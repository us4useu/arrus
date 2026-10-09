# NdArray + Lazy Operation Graph for ARRUS — Programming Plan Summary

This is the working summary of what we agreed to build. Full design
rationale lives in `ndarray_design.md`; the metamodel and figures live
under `figures/`. This document is meant to be readable in one sitting.

---

## 1. Goal

Add a NumPy-like n-d array interface to ARRUS whose operations are
**traced into an operation graph**, not executed eagerly. The graph is:

- **Serializable** to disk (protobuf) and reloadable.
- **Portable across backends** (CPU/numpy, CUDA/cupy, and — via the
  same interface — the **ultrasound device itself**).
- **Dynamically updatable at runtime** — variables re-bindable in the
  hot path; structural changes yield cheap new graph revisions.
- **Interoperable** with JAX (via `jax.make_jaxpr`) and PyTorch (via
  `torch.fx`) for the processing subset.
- **MLIR-shaped** so a later port to MLIR is mechanical rather than a
  rewrite.

User surface: `import arrus.numpy as anp`.

---

## 2. Core concepts

| Concept        | Role                                                                                                  |
|----------------|-------------------------------------------------------------------------------------------------------|
| `NdArray`      | **Lazy graph node** (traced value). Wraps an SSA `Value`. `arrus::framework`.                          |
| `NdStorage`    | **Concrete memory** returned by evaluation. Renamed from the old `NdArray`. `arrus::framework`.        |
| `NdArrayType`  | `(shape, dtype, device)` type descriptor. Shared by `NdArray` (may have unknown dims) and `NdStorage`. |
| `DType`        | Enum of numeric types; values are stable on-disk wire codes.                                           |
| `DeviceRef`    | Placement string: `"ANY"` (default), `"CPU"`, `"GPU:0"`, `"Us4R:0"`, `"Us4R:0/OEM:2"`, …               |
| `Value`        | SSA result of an `Op` — a lightweight `(Op*, index)` handle.                                           |
| `Op`           | Immutable node: `(name, operands, attrs, result_types)`. Non-copyable, non-movable.                    |
| `Attribute`    | Immutable typed payload on an Op (int/float/bool/string in v1).                                        |
| `Variable`     | Op producing a Value whose data is resolved from a runtime binding table.                              |
| `Graph`        | Immutable, persistent container of Ops + inputs/outputs/variables. Supports `copy_with(edits)`.        |
| `GraphHistory` | Versioned chain of graph revisions; serializable, replayable.                                          |
| `OpRegistry`   | Per-op entry: signature, verifier, shape/type inference, folder, canonicalizations, `apply_edit`.      |
| `Backend`      | Dispatches ops by name to concrete kernels. CPU, CUDA, and Us4R are backends.                          |
| `Executor`     | Topo-walks the graph, resolves placement, binds variables, moves data across backends.                 |

**Never `Tensor`** anywhere in ARRUS. Names are `NdArray*`.

---

## 3. Two-level compilation

The IR is **not two separate layers** — it is one graph in which
composite ops nest smaller ops (MLIR "regions-inside-ops" pattern).

- **Level 1 — `arrus` dialect** (device-agnostic).
  Examples: `arrus.fft`, `arrus.tx_rx`, `arrus.compute_delays`,
  `arrus.compose_tx`, `arrus.acquire`.
- **Level 2 — device dialects** (device-specific).
  Examples for the ultrasound device: `us4r.delays_to_cycles`,
  `us4r.set_delays_register`, `us4r.set_aperture_mask`,
  `us4r.arm_trigger`, `us4r.fire`.
  Analogous families for `cpu.*` and `gpu.*` (cupy-backed).

`session.compile(graph, target)` runs each backend's `Backend.compile`,
which **lowers** a Level-1 composite op into a body of Level-2 ops from
its own dialect. The pre-lowering and post-lowering forms coexist as
nested scopes in the same graph.

---

## 4. Mutability model

Structurally immutable graph with **two orthogonal update mechanisms**.

### A. Variables — runtime-rebindable, no structural change

- `Variable("sos", …)` is a first-class graph node with a mutable
  binding in the executor.
- Consumers re-read on next execute. Cross-backend consumers (e.g.,
  beamformer on GPU + `arrus.tx_rx` on Us4R) see a coherent value.
- Use when the value does **not** affect any output shape or dtype
  (speed of sound, TGC gains, thresholds, small parameter vectors).
- Hot-path update: `session.set_variable("sos", 1500.0)` — no graph
  rewrite, no reshape.

### B. Structural edits — new revision via `copy_with`

- `new_graph = old_graph.copy_with(edits={"depth": 60})`.
- Persistent-data-structure copy: unaffected sub-DAGs are shared by
  reference; only rewritten ops (and downstream shape-inference
  results) materialize.
- Use when shapes/dtypes/connectivity/op selection change (image
  depth, sample count, sequence length).
- **Rewrite discipline (SSA):** passes never mutate op operands or
  attrs in place; they build new ops and call
  `value.replace_all_uses_with(new_value)`, then a DCE sweep cleans
  up. This mirrors MLIR's rewriter and is what makes a future MLIR
  port mechanical.

### C. `GraphHistory` — versioned chain

- Serializable ordered chain of `(edit, resulting_graph)` pairs, each
  optionally stored as a delta.
- `history.checkout(revision_id)` yields the graph at that revision.
- Enables reproducible replay and V&V audit trails.

---

## 5. Dynamic ultrasound scheme update

Two swap tiers, both barrier-atomic.

- **Warm-path (variable rebind).** Update happens **between TX/RXs**.
  Backed by a **double-buffered device shadow** (Us4R DDR): compute
  the new register values into the shadow buffer while the current
  TX/RX plays out; at the trigger boundary the register set flips.
  No graph rewrite. Example: SoS change → new delays → new cycles →
  shadow write → flip.
- **Cold-path (structural rewrite).** Update happens **between
  sequences**. `copy_with` produces a new graph revision, the
  executor recompiles from it, and the next sequence uses the new
  layout. Example: image depth change.

---

## 6. Frame provenance and async consistency

The ultrasound device produces frames asynchronously; the graph and
its variables can change while frames are in flight.

Each frame carries provenance stamped by the backend:

- `revision_id` — which graph revision produced this frame.
- `variable_epoch` — a monotone counter of variable bindings visible
  when the frame was fired.

Four supported consumer patterns:

1. **Latest-wins** (default): the executor pulls the most recent
   frame and discards older buffered frames.
2. **Frame-matched**: two consumers on the same frame are aligned by
   `frame_id`; downstream ops see mutually consistent data.
3. **Discard-stale**: reject any frame whose `variable_epoch` is
   older than the current one.
4. **Revision-filter**: only accept frames whose `revision_id`
   matches the currently compiled graph.

---

## 7. Feedback loops via `store_variable`

Feedback (e.g., closed-loop TGC, adaptive gain) closes through the
runtime binding table — **not** through an IR back-edge — so the IR
stays a DAG (SSA-preserving).

- `store_variable(v, value_op)` — a side-effecting sink op that
  writes `value_op`'s output into `Variable(v)`'s next-epoch binding
  at the end of the current frame.
- On the following frame the downstream ops read the updated value.
- Sequenced side-effects; no edges from a producer back to a
  producer.

---

## 8. Backends

Abstract `Backend` interface with `dispatch(op, inputs, attrs,
variable_bindings) -> outputs`, plus per-backend `compile(graph)` for
lowering Level-1 composite ops.

Shipped:

- **CPU** (numpy kernels) — always available.
- **CUDA** (cupy kernels) — optional import, cross-backend numeric
  parity tests.
- **Us4R** — ultrasound device as a backend. Owns the `us4r.*`
  dialect and DDR shadow buffer.

Out-of-tree backends (ROCm, oneAPI, custom FPGAs) register their own
op implementations against the same registry.

Device selection is caller-driven. Ops in the IR are
backend-neutral (`arrus.fft`, not `arrus.fft.cuda`); users specify
targets at `session.compile(graph, target=...)`.

---

## 9. Op set

### v1 — processing (device-agnostic)

| Category           | Ops                                                                 |
|--------------------|---------------------------------------------------------------------|
| Factory            | `zeros`, `ones`, `full`, `empty`, `arange`, `linspace`, `array`     |
| Elementwise arith  | `+`, `-`, `*`, `/`, `//`, `%`, `**`, `neg`, `abs`                   |
| Elementwise math   | `exp`, `log`, `log2`, `sqrt`, `sin`, `cos`, `tan`, `atan2`          |
| Comparison         | `==`, `!=`, `<`, `<=`, `>`, `>=`, `where`                           |
| Reductions         | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `prod`, `any`, `all` |
| Shape              | `reshape`, `transpose`, `swapaxes`, `squeeze`, `expand_dims`,        |
|                    | `broadcast_to`, `concatenate`, `stack`                              |
| Indexing           | `__getitem__` (slice, integer, boolean mask, gather)                |
| Type/device        | `astype`, `to_device`                                               |
| FFT                | `fft`, `ifft`, `fftshift`, `ifftshift`                              |
| Complex            | `real`, `imag`, `conj`, `angle`                                     |
| Ultrasound         | `iq_demod`, `hilbert`, `matched_filter`                             |

### v1.1 — acquisition (Us4RBackend)

| Category     | Ops                                                             |
|--------------|-----------------------------------------------------------------|
| Acquisition  | `arrus.tx`, `arrus.rx`, `arrus.tx_rx`, `arrus.sequence`         |

Every op has a signature, a shape/type-inference rule, at least one
backend impl, a unit test, and a cross-backend equivalence test.

---

## 10. Serialization

Protobuf schemas alongside the existing `arrus/core/io/proto/`:

- `ndgraph.proto` — `NdGraph`, `Value`, `Op`, `Variable`, `Attribute`,
  `NdArrayType`, `NdArrayData`.
- `GraphHistory` — ordered `Revision` chain, either full snapshot or
  delta.

Files carry a magic header + `abi_version`. Loader refuses
unknown-major files with a clear diagnostic (same policy as the
existing ARRUS adapter ABI).

---

## 11. Interop

Optional imports, per adapter:

- `arrus.numpy.interop.jax`: `from_jax(jaxpr) ↔ to_jax(graph)`.
- `arrus.numpy.interop.torch`: `from_torch_fx ↔ to_torch_fx`.

Both explicitly list the ops they can round-trip; unmapped ops raise
`UnsupportedOpError` naming the missing entry.

---

## 12. Bridge to existing `arrus.utils.imaging.Pipeline`

Two additive changes, no removals:

1. `Operation.to_ndgraph(input_types) -> arrus.Graph`
   — default raises `NotImplementedError`. Ops that migrate override
   this to emit an equivalent subgraph.
2. `NdGraphOperation(Operation)` — wraps a saved `.ndgraph` file so
   an existing `Pipeline` can execute a saved graph as one step.

Users adopt NdArray graphs incrementally without rewriting whole
pipelines.

---

## 13. Directory layout (new artifacts)

```
arrus/core/api/framework/
  DType.h                 [Phase 2a — DONE]
  DeviceRef.h             [Phase 2a — DONE]
  Shape.h                 [Phase 2a — DONE]
  NdArrayType.h           [Phase 2a — DONE]
  Attribute.h             [Phase 2b — DONE]
  Op.h        (Value+Op)  [Phase 2b — DONE]
  NdStorage.h             [Phase 1  — DONE, renamed from NdArray.h]
arrus/core/api/graph/     [Phase 2c+]
  Graph.h, GraphHistory.h, OpRegistry.h, Backend.h, Executor.h, ...
arrus/core/graph/         [Phase 2c+]
  Graph.cpp, Rewriter.cpp, passes/, ops/, backends/
arrus/core/framework/
  NdArrayTypeTest.cpp     [Phase 2a — DONE]
  OpTest.cpp              [Phase 2b — DONE]
arrus/core/io/proto/
  ndgraph.proto           [Phase 3]
api/python/wrappers/ndgraph/   [Phase 6]
api/python/arrus/numpy/        [Phase 7]
```

---

## 14. Implementation phases

| Phase | Scope                                                                                | Status |
|-------|--------------------------------------------------------------------------------------|--------|
| 1     | Rename C++ `NdArray → NdStorage`; namespace `arrus::framework`. Design doc.          | DONE   |
| 2a    | Type descriptors: `DType`, `DeviceRef`, `Shape`, `NdArrayType`. GTest coverage.       | DONE   |
| 2b    | SSA nodes: `Attribute`, `Value`, `Op`. GTest coverage.                                | DONE   |
| 2c    | `Graph` (immutable, persistent), uses index, builder API.                             | next   |
| 2d    | `OpRegistry` + verifier (signature, shape/type inference).                            |        |
| 2e    | `Rewriter`: `copy_with`, `replace_all_uses_with`, DCE.                                |        |
| 2f    | Base ops (factory + elementwise + shape). Enough to author a trivial graph.           |        |
| 3     | Proto schema + `NdGraphSerializer` (Graph + GraphHistory round-trip).                 |        |
| 4     | `Backend` interface + `CpuBackend` + first ops.                                       |        |
| 5     | `Executor` + shape/type-inference pass + constant folding. End-to-end on CPU.         |        |
| 6     | pybind11 module `arrus._ndgraph`.                                                     |        |
| 7     | Python frontend `arrus.numpy`: `NdArray`, factories, ops, default-graph context.      |        |
| 8     | CUDA backend + cross-backend numeric parity tests.                                    |        |
| 9     | FFT, complex, ultrasound ops on both backends.                                        |        |
| 10    | Bridge to existing `Pipeline`: `to_ndgraph()` + `NdGraphOperation`.                   |        |
| 11    | Interop adapters (JAX, PyTorch), guarded by optional imports.                         |        |
| 12    | Docs (`api.python.rst`, tutorial).                                                    |        |
| 13    | v1.1 — Acquisition op family + `Us4RBackend`. Unified acquisition+processing graph.   |        |

---

## 15. Out of scope for v1

- Autograd / reverse-mode differentiation in the IR.
- Control-flow ops (loops, conditionals, scan). Regions are designed
  for but not populated in v1.
- Full numpy API coverage.
- Actual MLIR migration (dialect definition, LLVM/CUDA lowering) —
  the IR is shaped so this becomes mechanical when the need arises.
- Distributed / multi-GPU execution.

---

## 16. Verification strategy

- Unit tests per op per backend.
- Cross-backend equivalence (CPU vs CUDA) within numeric tolerance;
  skipped if cupy absent.
- Serialization round-trip on non-trivial graphs and `GraphHistory`
  with multiple revisions.
- Variable-update test: same variable feeding two backends;
  `set_variable` visible to both without a rewrite.
- `copy_with` test: shape propagation correct; un-edited sub-DAG
  shared by identity.
- Interop round-trip per supported op (JAX and PyTorch), skipped if
  dep absent.
- Bridge regression: existing imaging tests must still pass after the
  pilot op migrates.
- v1.1 unified test: `arrus.tx_rx` (Us4R) + processing ops (CUDA)
  sharing `Variable("sos")`; runtime update affects both without a
  rewrite.
- Jenkins picks the new tests up automatically once they land under
  the existing `*/tests/` convention.
