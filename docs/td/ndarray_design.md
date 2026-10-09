# NdArray + Operation Graph — Design

Status: **Draft**. Companion to the "NdArray" section of [requirements.md](requirements.md).

## 1. Goals

ARRUS introduces an `NdArray` type whose operations are **traced** into a graph rather than executed eagerly. The graph:

- describes computation on an **abstract** (device-agnostic) target,
- is serializable to and loadable from disk,
- runs on any registered backend (CPU, CUDA, later FPGA / ROCm / oneAPI / the ultrasound device itself),
- is usable from both **Python** (`import arrus.numpy as anp`) and **C++**,
- converts to and from **JAX** (JAXPR) and **PyTorch** (`torch.fx` / `torch.export`) graphs, for the processing subset,
- supports **runtime updates**: mutable Variables re-bound without rewriting the graph; cheap structural copies via `copy_with`; a serializable revision history.

Non-goals for v1: autograd, control-flow ops (loops/conditionals), full NumPy API coverage, distributed / multi-GPU execution, actual MLIR migration.

## 2. Nomenclature

| Term | Meaning |
|------|---------|
| `NdArray` | Lazy, traced n-d array. Wraps an SSA `Value` in the graph. Lives in `arrus::framework::NdArray` on the C++ side; exposed in Python as `arrus.numpy.NdArray` (i.e., `import arrus.numpy as anp` re-exports the framework class). What users write against in `arrus.numpy`. |
| `NdStorage` | Concrete memory-holder. Gives direct access to underlying memory (host RAM pointer, GPU device pointer, ...). Returned by evaluation — when the user runs a graph or a single op and wants plain bytes at an addressable location. Lives in `arrus::framework::NdStorage` (same namespace as `NdArray`). The type currently named `NdArray` in `arrus/core/api/framework/` is being renamed to `NdStorage` in Phase 1 to reflect this role and to free the `NdArray` name for the lazy type. Semantics match PyTorch's `Storage`: "the concrete memory backing an array." |
| `NdArrayType` | `(shape, dtype, device)` triple. **Shared** between `NdArray` and `NdStorage`: both carry an `NdArrayType`. On `NdArray` the shape may contain unknown dims (unresolved shape inference); on `NdStorage` all dims are concrete. One type descriptor, one serialization schema, one validation code path. |
| `Stream[T]` | A value type representing a possibly-infinite ordered sequence of `T` values. Trigger events and per-frame acquisition data both flow as streams. |
| `Value` | Immutable SSA result of an `Op`. |
| `Op` | Immutable `(name, operands, attrs, result_types)` node. |
| `Graph` | Immutable container of Values, Ops, inputs, outputs, Variables. |
| `Variable` | Runtime-mutable named cell; consumed by ops as an operand. |
| `GraphHistory` | Ordered chain of graph revisions (snapshot + deltas). |
| `Backend` | Concrete executor for a set of ops on a specific device (CPU, CUDA, Us4R, ...). |
| `Executor` | Walks a compiled graph, dispatches ops to backends, moves data across boundaries. |
| **Trigger source op** | An op with no data operands that produces a `Stream[Event]`. Roots of trigger flow. Kinds: `arrus.trigger.internal(prf=...)`, `arrus.trigger.external(port=...)`, `arrus.trigger.software(name=...)`. |
| **Trigger** | A single event on a trigger stream. Causes one fire of the ops that consume it. |
| **Trigger region** | The set of ops reachable from a given root trigger-source node. Emergent from graph structure; not a first-class user concept. |

**Do not use "Tensor" anywhere.** Naming convention is `NdArray*` throughout.

## 3. Type system

```
NdArrayType(shape: Tuple[Optional[int], ...], dtype: DType, device: DeviceRef)
Stream[T]       # ordered sequence of T; T is typically NdArrayType or Event
Event           # unit-typed value carried on trigger streams (may carry timestamp)
```

- `shape` may contain `None` (unknown) until shape inference runs.
- `DType` enum: `bool`, `int16`, `uint8`, `int32`, `int64`, `float32`, `float64`, `complex64`, `complex128`.
- `DeviceRef` is a string-shaped abstract placement: `"ANY"` (default), `"CPU"`, `"GPU:0"`, `"Us4R:0"`, ...
- **Ops are device-agnostic** in the IR. Placement is resolved at compile time by the session/executor, not baked into op names.
- **Scalars vs streams.** A scalar `NdArray` value (including a Variable's result) is treated as a constant stream by any consuming op — every fire of a downstream op observes the current value. This makes scalar ↔ stream interaction rule-based and matches how the executor actually implements it (consumers re-read the Variable binding on every fire).

## 4. Graph model (MLIR-shaped, without the MLIR dependency)

Discipline is enforced from day one so a future MLIR migration is mechanical:

- **SSA values.** Every `Op` produces immutable `Value`s with stable identity.
- **Explicit def-use.** Each `Op` knows its operand values; the `Graph` owns a lightweight uses index.
- **Immutable ops.** Rewrites produce **new** ops; no in-place mutation. Rewriter calls `value.replace_all_uses_with(new_value)` and lets DCE clean up.
- **Op registry.** One entry per op: signature (arity, type constraints), shape/type inference rule, optional constant folder, optional canonicalization patterns, optional `apply_edit`.
- **Plain-data attributes.** Numbers, strings, bools, small `NdArrayData` blobs, lists of same. No closures, no live Python objects.
- **Persistent-data-structure Graph.** Copies share unaffected sub-DAGs by reference (see §7).
- **Default-graph stack.** Thread-local; `with arrus.Graph() as g:` pushes onto it. Factory functions (`anp.zeros`, ...) attach to the current graph.
- **Firing semantics.** The unit of firing is the **trigger region**, not the individual op. A region fires once per new element arriving on its input stream(s). When a region fires, all its ops execute in topological order; ops inside the region do *not* individually wake up on their internal edges. Scalar operands (Values) are re-read on every fire. Trigger sources are ops that produce `Stream[Event]` without data operands; they are the roots of firing flow. **Trigger regions are emergent** from graph structure — the set of ops reachable from a given root trigger-source op — and do not need to be declared by the user.
- **Runtime realization of `Stream[T]` edges.** Same IR type, two runtime shapes chosen by the compiler:
  - **Queue realization** — real ring buffer + notification + per-consumer read cursor. Used at region boundaries (Us4R → GPU, per-frame region → per-N-frame region, cross-device).
  - **Shared-buffer realization** — a single output buffer, no queue, no notification. Used inside a region: the region-level wake-up already covers all internal ops; kernel/stream ordering enforces sequencing. This is how existing `ProcessingRunner` already dispatches intra-region GPU chains (`arrus/utils/imaging.py:369`).

  The IR level (`Stream[T]`) is uniform; the runtime realization is a compilation-time decision, not part of the type.
- **Side-effecting sinks.** A small number of ops produce no SSA results and exist for their side effects (writing runtime state, logging, etc.); `arrus.store_variable(name, value)` is the canonical example (see §5.1 and §7.10). They participate in the topological sort via their *operand* dependencies, execute as leaves, and are treated by the verifier as legal sinks — they do not violate SSA (they produce no values to name) and they do not introduce cycles (they close feedback loops through the runtime binding table, not through IR edges).

### 4.1 Graph metamodel

The metamodel — what kinds of nodes exist, what kinds of edges connect them, and what constraints apply — is declared in a single YAML file that is the source of truth for both the diagram and the runtime verifier.

- **Source of truth:** [`figures/ndarray_metamodel.yaml`](figures/ndarray_metamodel.yaml). Six node kinds (`Op` abstract, `Variable` and `TriggerSource` extending `Op`, plus `Value`, `Device`, `Stream`), seven edge kinds, ten global constraints.
- **Diagram:** [`figures/ndarray_metamodel.png`](figures/ndarray_metamodel.png) (auto-generated by [`figures/generate_metamodel_diagram.py`](figures/generate_metamodel_diagram.py)). UML-style class diagram with generalization arrows for `extends` and vee arrows for associations, labelled with edge names and endpoint cardinalities.
- **Rationale for staying custom (vs. RDF+SHACL / OWL):** the metamodel is small (6×7), the validator runs *inside* the graph library on every construction and rewrite, and the team is Python-first. YAML+codegen keeps one source of truth for the diagram and the validator without pulling in semantic-web infrastructure. If we ever need to publish the metamodel as an RDF shape, a SHACL emitter alongside the diagram generator is a mechanical addition.

### 4.1.1 Composite Ops and Devices

Both Ops and Devices are composable: an Op can be a subgraph of Ops (e.g., `bmode_pipeline` composed of `remap → transpose → bandpass → ...`), and a Device can contain sub-Devices (e.g., `Us4R:0` contains `OEM:0..N`, each of which contains `Channel:0..M`).

The metamodel encodes this with two self-loop edges:

- `contains_op` (`Op → Op`, `0..* : 0..1`) — a composite Op contains child Ops. An Op with zero outgoing `contains_op` edges is a leaf; one with N is a composite. Root Ops (no incoming `contains_op`) live directly in the Graph.
- `contains_device` (`Device → Device`, `0..* : 0..1`) — a composite Device contains sub-Devices.

Composites are **first-class nodes**, not implicit naming prefixes. That lets each composite carry its own attributes, interface, documentation, and validation contracts — a composite Op has a declared external interface (types of its boundary operands and results), a composite Device has capabilities and health state, and a saved `.ndgraph` can carry composites as reusable units.

**Composite interfaces.** A composite Op's external interface is declared by two boundary marker ops inside its sub-DAG: `arrus.io.input` (each declares one external operand slot) and `arrus.io.output` (each declares one external result slot). The `composite_op_interface_matches_boundary` global constraint enforces that the composite's declared operand/result types match the marker types.

**Placement into composite Devices.** An Op placed on a composite Device (e.g., `Us4R:0`) is resolved at compile time by the compiler to a leaf Device inside it (`Us4R:0/OEM:2`). This is what makes composite devices useful as placement targets — the user writes intent at a high level, the compiler picks the concrete leaf. The `placement_within_containment` global constraint enforces the resolution.

**Path-like IDs are derived, not stored.** Every node's `qualified_id()` is computed by walking its containment parents up to the root:

```
Us4R:0/OEM:2/Channel:5
imaging_pipeline/bmode/rxbf_lin
```

Users can look up any descendant by path (`graph.find("Us4R:0/OEM:2/Channel:5")`) — same ergonomics as pure path IDs, none of the fragility. Renaming a composite is local: the composite's `local_name` changes, and all descendants' qualified IDs update automatically. A working demonstration lives at [`figures/qualified_id_demo.py`](figures/qualified_id_demo.py).

## 5. Op set

Ops in ARRUS belong to **dialects**, each identified by a namespace prefix. The tables below are for the `arrus` dialect (device-agnostic, Level-1 authoring). Backends contribute their own dialects (`us4r.*`, `cuda.*`, `cpu.*`, ...) — those Level-2 ops are produced by lowering and not authored directly by users; they are described in each backend's own reference (see §6.1 for the two-level model).

### 5.1 v1 — processing (device-agnostic op names)

| Category | Ops |
|----------|-----|
| Factory | `zeros`, `ones`, `full`, `empty`, `arange`, `linspace`, `array` |
| Elementwise arith | `+`, `-`, `*`, `/`, `//`, `%`, `**`, `neg`, `abs` |
| Elementwise math | `exp`, `log`, `log2`, `sqrt`, `sin`, `cos`, `tan`, `atan2` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=`, `where` |
| Reductions | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `prod`, `any`, `all` |
| Shape | `reshape`, `transpose`, `swapaxes`, `squeeze`, `expand_dims`, `broadcast_to`, `concatenate`, `stack` |
| Indexing | `__getitem__` (slice, integer, boolean-mask, gather) |
| Type/device | `astype`, `to_device` |
| FFT | `fft`, `ifft`, `fftshift`, `ifftshift` |
| Complex | `real`, `imag`, `conj`, `angle` |
| Ultrasound | `iq_demod`, `hilbert`, `matched_filter` |
| Control / state | `store_variable(name, value)` — side-effecting sink that writes `value` into the runtime binding table for the named Variable. Enables in-graph feedback loops (§7.10). No SSA result; not consumed by anything downstream. |

### 5.2 v1.1 — acquisition (`Us4RBackend`)

| Category | Ops |
|----------|-----|
| Trigger sources | `arrus.trigger.internal(prf=...)`, `arrus.trigger.external(port=...)`, `arrus.trigger.software(name=...)` |
| Delay computation | `arrus.compute_delays(sos, geometry, aperture, focal_points)` |
| Descriptor assembly | `arrus.compose_tx(aperture, pulse, delays)`, `arrus.compose_rx(aperture, sampling_config)` |
| Acquisition | `arrus.tx_rx(tx_desc, rx_desc, trigger)`, `arrus.sequence(tx_rxs)`, `arrus.acquire(sequence)` |

The acquisition op family is deliberately **decomposed** so that runtime updates (see §7) affect the smallest possible sub-DAG. In particular, TX delays are a separate op — a `set_variable("sos", ...)` update reaches only `compute_delays` and its downstream, not the whole `tx_rx` descriptor.

Every op ships with: signature, shape/type inference rule, at least one backend implementation, a unit test, and a cross-backend equivalence test where applicable.

## 6. Backends

- **`Backend` interface:** uniform across all devices, regardless of whether they execute autonomously or on host demand.
  - `upload(op, attrs, bindings) -> handle` — called at compile time; materializes resources (device memory allocation, descriptor upload, kernel selection).
  - `dispatch(handle, input_streams, bindings)` — called when the op's containing trigger region fires; produces output stream elements.
  - `update(handle, changed_fields, new_values)` — called on `set_variable` when the change reaches an already-uploaded op; issues a partial rewrite (see §7).
  - `start(handle)` / `stop(handle)` — for ops whose trigger source is autonomous (e.g., `arrus.trigger.internal`); the backend arms / disarms hardware to begin / cease firing.
- **Shipped in v1:** `CpuBackend` (numpy), `CudaBackend` (cupy).
- **Shipped in v1.1:** `Us4RBackend` for acquisition ops.
- **Out-of-tree:** ROCm, oneAPI, custom FPGA backends register themselves via a Python or C++ entry point; the ARRUS core does not need to change.
- **Placement resolution:** the user selects targets at `session.compile(graph, target=...)`; `target` may be a single string (`"GPU:0"`) or a map keyed by op family / role (`{"acquisition": "Us4R:0", "processing": "GPU:0"}`) for graphs mixing acquisition and processing. Ops with `device == "ANY"` flow to whichever backend the session dispatches them to; ops with a concrete `device` hint must be honored (or the compile fails).
- **Cross-backend transfers:** when consecutive ops resolve to different backends, the executor inserts implicit copies via typed stream buffers. Copies are minimized by a placement-planning pass.
- **On-the-fly partial rewrites.** `Us4RBackend` uploaded descriptors are stored in **double-buffered device memory**: at any time one copy is live for the running TX/RX loop, and the other is available for host-side rewrites. Partial rewrites (from `update(handle, changed_fields, new_values)`) go to the shadow copy; a barrier flips the buffers at a safe point between TX/RXs. Acquisition never pauses. This is what keeps `set_variable` updates that touch acquisition-side descriptors (see §7.2) non-disruptive — regardless of dirty-set size.

### 6.1 Two-level compilation and the machine-code graph

The `Backend` interface operates on **two levels of IR**, not one. This is the same shape every mature ML compiler has (LLVM+MLIR, XLA, IREE, Triton) and it's what makes ARRUS's dynamic-scheme-update (§7) both cheap and precise.

- **Level 1 — device-agnostic authored graph.** Uses ops from the `arrus.*` dialect: `arrus.tx_rx`, `arrus.beamform`, `arrus.compute_delays`, `arrus.iq_demod`, ... This is what users write in `arrus.numpy`. Devices may be `"ANY"` throughout; placement is resolved by the compiler.
- **Level 2 — device-specific "machine-code graph".** Uses ops from backend-specific dialects: `us4r.set_delays`, `us4r.set_aperture`, `us4r.arm_trigger`, `us4r.fire`, `cuda.launch_kernel`, `cpu.numpy_call`, ... One dialect per backend. These are the actual primitives the device understands.

Each `Backend` implements a `compile` method that lowers Level-1 ops assigned to that backend into an equivalent Level-2 subgraph:

```cpp
Graph Backend::compile(const Graph& high_level_subgraph);
```

Lowering is a **rewrite pass**, expressed via the same `Rewriter` used elsewhere: walk the subgraph, match Level-1 ops, and produce Level-2 replacements via `replace_all_uses_with`. Backends are dialect authors — they define their Level-2 ops (registered with the `OpRegistry`) and the lowering rules from Level 1 → their dialect.

### 6.1.1 The full compilation pipeline

```
User graph (Level 1)                              arrus.tx_rx, arrus.beamform, ...
    │
    │   Level-1 passes (device-agnostic):
    │     shape/type inference, constant folding, DCE, copy_with restructuring
    │
    │   Placement resolution: assign each Level-1 op to a backend
    │
    │   For each backend: Backend.compile(subgraph) → Level-2 subgraph
    ▼
Mixed graph (Level 2)                             us4r.set_delays, cuda.launch_kernel, ...
    │
    │   Level-2 passes (ARRUS-driven, backend-neutral machinery):
    │     consecutive-write fusion, PCIe-transaction reordering,
    │     per-Level-2-op dirty-set precomputation for each Variable
    │
    │   For each backend: Backend.upload(level2_subgraph) → device state
    ▼
Compiled program ready to start()
```

Both levels live in the same `Graph` type; only the op names (and thus the registered signatures/backends) differ. A `Graph` can hold Level-1 and Level-2 ops simultaneously during the lowering process.

### 6.1.2 Why the two-level split matters

- **Level 1 is authoring-friendly.** Users don't think about set-delays byte layout or PCIe transactions; they think in `tx_rx`, `beamform`, `compute_delays`.
- **Level 2 is where optimization pays off.** Post-lowering, ARRUS sees the actual device primitives and can fuse consecutive `us4r.set_*` writes to overlapping memory, reorder PCIe transactions, and compute dirty sets at the exact byte-region granularity.
- **Dirty-set analysis at Level 2 is dramatically finer than at Level 1.** At Level 1, an SoS change dirties `arrus.tx_rx` — a monolithic op. At Level 2, the same change dirties only `us4r.set_delays`; `us4r.set_aperture`, `us4r.arm_trigger`, and `us4r.fire` are bit-identical and their device memory is untouched. **This is the mechanism that delivers the "minimal propagation of changes to the device" property.**
- **Backends stay swappable.** A new hardware family adds a new Level-2 dialect + a new `Backend.compile` implementation; nothing at Level 1 changes. Same authored graph, different backend = different target.

### 6.1.3 Concrete example: SoS change through both levels

**Level 1 (as authored):**

```python
sos    = arrus.Variable("sos", 1540.0)
delays = arrus.compute_delays(sos, geometry, aperture, focal_points)
tx     = arrus.compose_tx(aperture, pulse, delays)
tx_rx  = arrus.tx_rx(tx, rx, trigger=clk)
```

Level-1 dirty-set for `sos`: `{compute_delays, compose_tx, tx_rx}` — coarse.

**Level 2 (after `Us4RBackend.compile`):**

```
us4r.set_aperture(mask=<from aperture>)                 // does NOT depend on sos
us4r.set_pulse_shape(id=<from pulse>)                   // does NOT depend on sos
us4r.set_delays(bytes=<computed from sos>)              // depends on sos
us4r.arm_trigger(source=internal, prf=<from clk>)       // does NOT depend on sos
us4r.fire()                                             // does NOT depend on sos
```

Level-2 dirty-set for `sos`: `{us4r.set_delays}` — precise.

**Runtime `set_variable("sos", 1500.0)`:**

1. Look up `compiled.dependents_of["sos"]` (precomputed against the *Level-2* graph): `[us4r.set_delays#0, us4r.set_delays#1, ..., gpu.kernel_arg("sos_beamformer")]`.
2. Re-run host-side derived-value ops (`compute_delays`) → new delay bytes.
3. `Us4RBackend.update(set_delays_handle, {"bytes": new_bytes})` → partial PCIe write of the delays region only.
4. `CudaBackend.update(kernel_handle, {"sos_arg": 1500.0f})` → kernel-arg update; next launch sees the new value.
5. Barrier flips the shadow buffer between TX/RXs. Acquisition never pauses.

Aperture, pulse, trigger, fire: all untouched. This is why the two-level split is not an ornament — it's *how* the update stays minimal.

### 6.1.4 Multi-dialect `OpRegistry`

Because ops come from multiple dialects (`arrus`, `us4r`, `cuda`, `cpu`, `rocm`, ...), the `OpRegistry` is a **map of `dialect_name → {op_name → signature}`**, not a flat map. Op names carry their dialect prefix (`arrus.fft`, `us4r.set_delays`). Backends register their dialect at load time. The verifier applies the same L1 metamodel rules (SSA, cardinality, containment) uniformly to ops from any dialect — backends don't get to skip verification.

This is the same design as MLIR's dialect system, and lowering passes correspond exactly to MLIR's `DialectConversionPass`.

## 7. Dynamic Ultrasound Scheme Update

An incremental, in-flight update mechanism for a running ultrasound scheme, comprising two mechanisms: **Variable rebinds** (`set_variable`) that leave the compiled program intact, and structural **`copy_with` edits** that produce a new graph revision. The graph is structurally immutable; both mechanisms preserve SSA discipline (see §13).

Both mechanisms interact with the fact that the ultrasound device runs asynchronously w.r.t. the host — data continuously produced by the device may reflect *any past* state of the graph, not the current state. Consistency between host-driven updates and device-driven data production is maintained via **frame provenance stamping** (§7.9) — every frame carries metadata identifying the revision + variable-epoch under which it was fired.

### 7.1 Two mechanisms, one continuous cost + one recompile

| Mechanism | Trigger | What changes | Compiled program? | Cost |
|-----------|---------|-------------|-------------------|------|
| **Variable rebind** | `set_variable` | Runtime data referenced by the graph; graph itself unchanged | Unchanged | Bounded and statically known — a continuous function of the dirty set. µs (single kernel-arg swap) to tens of ms (many device writes / large derived-tensor recomputation). See §7.2. |
| **Structural rewrite** | `copy_with(edits)` | The graph itself (shapes, dtypes, connectivity, op selection) | Invalidated → recompile | 10s of ms to seconds (recompile + full re-upload; some edits can be applied gap-free by the backend). See §7.4. |

The distinguishing question is: **does the change invalidate anything structural in the compiled program?** (shapes, dtypes, kernel selection, memory layout, op count, placement)
- **No** → Variable + `set_variable`. Whether the update also requires device-memory writes is a *cost* concern, not a mechanism switch — it's the same code path either way.
- **Yes** → attribute + `copy_with`.

**Note:** earlier drafts of this doc split Variable rebinds into "hot" and "warm" tiers based on whether the update touched device memory. That split has been removed. The mechanism is uniform; the cost varies continuously with the dirty set. Drawing a discrete line anywhere in that spectrum is arbitrary and misleading.

### 7.2 Variable rebind — bounded partial update

- `Variable` is a first-class op producing a `Value` whose data is read from a runtime binding table (not baked into the graph).
- Ops consume Variables as operands. Updating a Variable does **not** rewrite the graph — only the runtime binding changes.
- Variables can be updated **from outside the graph** via `compiled.set_variable(name, value)` / `compiled.set_variables({...})` (atomic batch), or **from within the graph** via the `arrus.store_variable(name, value)` side-effecting sink op (§7.10). Both paths land in the same binding table; both trigger the same partial-update protocol below.
- **Dirty set is precomputed at compile time.** `compiled.dependents_of[name] -> [op, ...]` is a static lookup filled during `session.compile`. `set_variable` is an O(1) table lookup, not a runtime BFS.
- Consumers re-read on their next fire. If multiple backends consume the same Variable (e.g., GPU beamformer + Us4R `compute_delays` reading `sos`), all see the new value coherently.

**Update protocol** (executor's steps for each `set_variable` call):

1. Look up the precomputed dirty set for the Variable.
2. Re-execute affected host-side derived-value ops (e.g., `compute_delays`).
3. For each dirty op resolved to a device backend, call `backend.update(handle, changed_fields, new_values)`. The backend computes a byte-level diff between old and new bytes and writes only the changed region into the shadow copy of the double-buffered device memory.
4. A barrier flips the buffers at a safe point (between TX/RXs for `Us4RBackend`). Acquisition never pauses.

Ops that use fixed values (or a different Variable) are not in the dirty set — their device memory is untouched, their bindings unchanged.

**Cost is a continuous function of the dirty set.** Statically knowable from the graph, so users can predict it without profiling. Representative examples:

| Variable | Dirty set | Work per fire | Approx cost |
|----------|-----------|---------------|-------------|
| Scalar threshold consumed as one GPU kernel arg | 1 op, no derived values | Update kernel arg | ~µs |
| SoS consumed only as a GPU beamformer scalar arg | 1 op | Update kernel arg | ~µs |
| SoS → `compute_delays` (host) → partial write to one Us4R descriptor | 2 ops | Small host arithmetic + ~KB PCIe write | ~ms |
| SoS → `compute_delays` for a 200-line `LinSequence` → 200 partial descriptor writes | ~200 ops | 200× partial writes | ~10s of ms |
| TGC curve → per-frame GPU multiplier + device-side TGC table | 2 ops | GPU array copy + partial device write | ~ms |
| Large lookup table consumed by many GPU ops + cross-backend transfer | many ops | Recompute + PCIe upload | ~10s of ms |

Same mechanism, different cost. Cost is bounded and known; the design levers in §7.6 (op decomposition, overallocation) exist to keep the dirty set small when it matters.

### 7.3 Cost inspection API

Since the dirty set and per-op work are known at compile time, the compiled graph exposes them:

```python
compiled.update_cost("sos")
# -> UpdateCost(
#      dirty_ops=[compute_delays#0, compose_tx#0, tx_rx#0, ..., beamform#0],
#      host_recompute_bytes=800,
#      device_write_bytes=1600,
#      cross_backend_transfer_bytes=0,
#      estimated_us=1500,
#    )

compiled.explain_update("sos")
# -> human-readable trace: which ops fire, what each will do, byte counts
```

This gives users a way to answer "is this Variable update going to be fast enough to call at 30 Hz?" without profiling, and to notice when a naive op decomposition has made an update more expensive than it needs to be. Also useful in tests to lock in performance regressions (e.g., "assert `compiled.update_cost('sos').device_write_bytes < 2 * n_tx * DELAY_BYTES`").

Estimates are budgets, not measurements. Backends provide `estimate_update_us(handle, changed_fields)` used to fill in the sum; they may return `None` if they cannot estimate for a given op, in which case the executor reports `estimated_us=None`.

### 7.4 `copy_with(edits)` — structural rewrite, new revision

- Anything that changes shapes, dtypes, connectivity, or op selection is a structural edit.
- `new_graph = old_graph.copy_with(edits={...})` returns a new revision. Cheap because the graph is a persistent data structure: unaffected sub-DAGs are shared by reference; only rewritten ops (and their downstream shape-inference results) materialize new nodes.
- Every op has an optional `apply_edit(op, edit) -> op'` hook in its registry entry (default: raise if the edit is targeted at the op). The rewriter walks reachable ops, applies matching edits, and re-runs shape/type inference.
- Consuming the new revision requires recompilation (`session.compile(new_graph, ...)`); the old `CompiledGraph` is stopped and released, or kept for history.

**Use for:** image depth, sample count, sequence length, dtype changes.

### 7.5 `GraphHistory`

- Ordered chain of revisions with a base graph + `(edit, resulting_graph)` entries.
- Serializable in the `.ndgraph` container as a `GraphHistory` message wrapping `Revision`s; each revision is either a full `NdGraph` snapshot or an `EditSet` delta against its parent (snapshots inserted periodically to keep `checkout` fast).
- `history.checkout(revision_id)` yields the graph at that point.
- Enables reproducible replay of parameter/scan changes and audit trails for V&V.

### 7.6 Op-decomposition and overallocation as design levers

Both levers exist to **keep the dirty set small** for parameters that will be updated frequently. Neither changes the update mechanism — they change how much work each update ends up doing.

**Op decomposition.** How ops are factored controls how tight the dirty set can be. Instead of a monolithic `arrus.tx_rx(sos, geometry, aperture, pulse, ...)`, decompose so SoS-dependent computation is isolated:

```python
sos    = arrus.Variable("sos", 1540.0)
delays = arrus.compute_delays(sos, geometry, aperture, focal_points)   # only this depends on sos
tx     = arrus.compose_tx(aperture, pulse, delays)                     # trivial combiner
tx_rx  = arrus.tx_rx(tx, rx, trigger=clk)
```

Now `set_variable("sos", 1500.0)`:
- BFS from `sos` marks `compute_delays`, `compose_tx`, `tx_rx` dirty. Fixed-delay TX/RXs (or TX/RXs using a different `sos` Variable) are untouched.
- Executor re-runs `compute_delays` on host (small vector arithmetic).
- `Us4RBackend.update(tx_rx.handle, changed_fields={"tx.delays": new_delays})` writes only the delays region to the shadow buffer.
- Next TX/RX cycle uses the new delays; the rest of the descriptor is bit-identical and never touched.

**Overallocation.** You can trade upfront device memory for keeping updates in the Variable-rebind lane (avoiding a `copy_with` recompile). Example: allocate a max-size aperture once and expose a `Variable("active_aperture_mask", bool[max])`; changing which elements are active becomes a rebind, not a recompile. Same trick works for scan sequence length (upload max length + active-length Variable), TGC curves, and pulse shapes (small on-device library + selector Variable). Design cost: more device memory occupied. Operational benefit: many operator-facing knobs become Variable rebinds rather than structural rewrites.

### 7.7 Author guideline

> If a parameter can change without changing any output shape/dtype, expose it as a **Variable operand**. Otherwise, expose it as an **attribute** and provide an `apply_edit` for the op. If the naive attribute is shape-changing but can be reformulated over a fixed-size overallocation, prefer the Variable form. If a Variable is expected to be updated frequently, decompose the consuming ops so the dirty set stays small (see §7.6).

### 7.8 Mapping to representative use cases

- **Speed-of-sound change.** `Variable("sos", float32, initial=1540.0)` consumed by `compute_delays` and by the GPU beamformer. `set_variable("sos", 1500.0)` → Variable rebind: recompute delays on host, partial device-memory rewrite of the delays region only, GPU beamformer sees the new scalar on next fire. No acquisition pause. Cost is bounded by the number of dependent TX/RXs (see §7.2 table).
- **Image depth change.** `new_g = g.copy_with(edits={"depth": 60})` → structural rewrite: shape inference propagates (acquisition op's sample count changes, downstream reshapes/reductions follow), `GraphHistory` records the transition, `Us4RBackend` reuploads the descriptor with the new sample count.
- **Aperture change with overallocation.** `Variable("active_aperture_mask", bool[128])`, allocated to the max-size descriptor at compile time → Variable rebind: `set_variable(...)` rewrites only the mask region in the shadow buffer. No recompile, no reshape.

### 7.9 Frame provenance and asynchronous consistency

The ultrasound device runs on its own clock (internal PRF) and produces data continuously and independently of the host. The only synchronization point between the device timeline and the host timeline is the "new data arrived" interrupt. This creates a fundamental consistency problem: **any given frame arriving at the host was produced under some past state of the graph, not the current state**. Naïvely treating incoming frames as if they reflect the current Variable bindings or graph revision leads to subtle correctness bugs.

**Design principle:** every frame produced by any backend carries a self-describing provenance record. Downstream code routes and filters frames based on this record; it never assumes "current state" applies to received data.

#### 7.9.1 The `FrameProvenance` record

Every frame emitted by any backend into a `Stream[NdArray]` is accompanied by:

```
FrameProvenance {
    revision_id:        uint64      // compiled-graph revision that fired the TX/RX
    variable_epoch:     uint64      // monotonic counter, bumped by every set_variable / store_variable
    tx_rx_ordinal:      uint32      // TX/RX index within the sequence
    sequence_ordinal:   uint64      // sequence-repetition counter
    device_timestamp:   uint64      // device clock at TX/RX firing (nanoseconds)
}
```

- `revision_id` and `variable_epoch` establish "what world state produced this frame".
- `tx_rx_ordinal` and `sequence_ordinal` locate the frame within the acquisition schedule.
- `device_timestamp` is the ground truth for time-based reasoning (Doppler wall-time, absolute timestamps).

Backends fill in the device-specific fields (`device_timestamp`, `tx_rx_ordinal`, `sequence_ordinal`); the executor fills in `revision_id` (from the currently-armed compiled graph on the device) and `variable_epoch` (from the atomic counter). See §6 for the extended `Backend` interface obligations.

#### 7.9.2 Warm-path consistency (Variable rebind)

Each `set_variable` (or in-graph `store_variable`, §7.10) bumps a monotonic counter — the **variable_epoch**. The barrier flip between shadow and live device memory is where the epoch takes effect on the device:

1. Host calls `set_variable("sos", 1500)`.
2. Executor: `epoch := epoch + 1`.
3. Warm-path partial rewrite (§7.2) writes new delays to shadow memory.
4. **At the next between-TX/RX safe point**, the device flips shadow → live.
5. From that TX/RX onward, frames are tagged `variable_epoch = new_epoch`. Frames fired *before* the flip retain the old epoch, even if they reach the host after the flip happened.

The barrier is atomic w.r.t. TX/RX boundaries: a single TX/RX cannot span the flip. Its samples are either all-old-delay or all-new-delay, never mixed. This is what the double-buffered device memory (§6) guarantees.

**Binding history.** The executor keeps a small ring of past bindings: `epoch → {name → value}`. Consumers that need to know "what SoS was in effect when this frame was fired?" can look up `session.get_variable_at_epoch(name, frame.variable_epoch)`. Default ring size is 64 epochs; sized to cover realistic pipeline latencies; configurable per Variable.

#### 7.9.3 Cold-path consistency (`copy_with` + recompile)

`session.compile(new_g, ...)` allocates a new `CompiledGraph` with a fresh `revision_id`. The old graph does not immediately die. Two policies, chosen at compile time:

- **`drain=True` (default).** The old `CompiledGraph` stays live for the buffer drain. Its consumers keep processing frames tagged with the old `revision_id` until the buffer is empty; then the old graph self-closes. The new graph starts fresh on its own logical stream; consumers on the new graph never see stale-revision frames.
- **`drain=False`.** Hard swap. Old-graph frames still in the buffer are discarded (with a log entry per drop). The old graph's consumers stop immediately.

The device-side swap is a single atomic register write (the descriptor pointer flips at a safe point). **The safe swap point for structural rewrites is *between sequences* (a full acquisition sequence completing), not just between TX/RXs.** This is stricter than the warm-path rebind's safe point (which is between TX/RXs), because the sequence shape itself may change. Pre- and post-swap TX/RXs cannot mix; the atomic swap guarantees this.

#### 7.9.4 The interrupt is the only cross-clock synchronization edge

Everything else in the system runs asynchronously:

- **Device** fires at PRF autonomously; frames land in the device ring buffer.
- **Compiler** runs on the host on its own thread; produces new `CompiledGraph`s asynchronously.
- **Variable rebinds** are host-thread transactions against the binding table + shadow memory.
- **GPU processing** runs on the CUDA stream, its own timeline.

The "new data arrived" IRQ is the *only* edge that crosses the device ↔ host clock-domain boundary. On IRQ:

1. Read the frame descriptor (payload pointer + `FrameProvenance`) from the device ring buffer.
2. Transfer to the host ring buffer.
3. Wake the trigger region rooted at the acquisition op.
4. Region fires; downstream sees the frame with provenance attached.

Graph consistency does **not** depend on locking the graph during device execution. It depends on **every observation of "what the device produced" being timestamped with the world state that produced it**.

#### 7.9.5 Consumer patterns for handling provenance

Downstream code has four options, chosen per-consumer:

- **Latest-wins (default for most consumers).** Ignore provenance for Variable-derived values; use the current binding. Simple; correct when the physical quantity being estimated changes slowly compared to the acquisition rate. For SoS, TGC, gain: a one-frame lag is clinically negligible.

- **Frame-matched (opt-in, strict correctness).** Consumer explicitly reads Variable values matched to the frame's epoch:
  ```python
  def custom_beamform_processor(frame):
      sos = session.get_variable_at_epoch("sos", frame.provenance.variable_epoch)
      return beamform(frame, sos=sos)
  ```
  Costs one dict lookup per frame; guarantees each frame is processed with the Variable state under which it was acquired. Right for research consumers that iterate on the frame ↔ Variable relationship (adaptive SoS estimation, self-consistency loops).

- **Discard-stale.** Discard frames whose `variable_epoch != current`. Wastes frames but avoids all consistency questions. Right for calibration workflows where discarding a few frames per update is acceptable.

- **Revision-filter.** Consumers on a specific pipeline branch discard frames whose `revision_id` doesn't match the branch's revision. Automatically true for `drain=True` (each pipeline consumes its own revision); needs to be explicit only for exotic multi-revision workflows.

#### 7.9.6 Buffer sizing for the worst case

The device-side ring buffer must be large enough to hold a **drain window**:

- **Recompile time** — realistic upper bound depends on graph size; on the order of 10–100 ms for large sequences.
- **PRF** — frames arrive at that rate.
- **Required capacity:** at least `PRF × T_recompile` frames.

Similarly the host-side ring buffer needs headroom for GPU processing latency plus a safety margin. Overflow policy: **drop-oldest, log the drop with its provenance**. Frame drops are visible in the log so pathological recompile / processing situations are diagnosable.

#### 7.9.7 What this rules out

- **Retroactive updates.** A `set_variable` cannot retroactively affect frames already in flight or in the buffer. Those frames carry the old epoch, forever. Applications that rely on "the value applies to frames I already have" are broken by design; the mechanism is transparent about this.
- **Free-lunch tight feedback.** "Frame N → adjust Variable → apply to frame N+1" requires at least one full acquisition + processing round-trip. Provenance-tagging doesn't shorten that latency; it just makes the round-trip's semantics correct.
- **Mid-TX/RX updates.** A rebind cannot land partway through a TX/RX. The atomic barrier is between TX/RXs, always.
- **Mid-sequence structural rewrites.** A `copy_with` recompile cannot apply partway through a sequence. The atomic swap is between sequences, always.

#### 7.9.8 Backend obligations

For a `Backend` implementation to participate in the consistency contract, it must:

1. Emit `FrameProvenance` alongside every frame it produces into a `Stream[NdArray]`. The backend fills in device-specific fields; the executor fills in `revision_id` and `variable_epoch` via the interface hooks below.
2. Support atomic swaps at the documented safe points: between TX/RXs for warm-path rebinds; between sequences for cold-path recompiles.
3. Report the currently-armed `revision_id` to the executor so provenance can be stamped correctly.

The `Backend` interface (§6) is extended with:

```cpp
virtual FrameProvenance current_provenance() const = 0;
virtual void set_active_revision(uint64_t revision_id) = 0;
```

Both are called by the executor around swap operations; `current_provenance()` provides the seed used to stamp each outgoing frame.

For pure-processing backends (`CpuBackend`, `CudaBackend`) that do not originate frames, provenance is *forwarded* — each op reads the provenance of its input frame and attaches the same provenance to its output. Processing preserves the acquisition-time identity of the data.

#### 7.9.9 What Phase-2 must reserve for this

Phase 2 (core IR) must reserve the following:

- Every op result of type `Stream[NdArray]` implicitly carries a per-element `FrameProvenance`. Phase 2's `Stream[T]` type must be able to represent this without expanding to `Stream[(T, FrameProvenance)]` at every use site — provenance is a companion band, not part of the payload type.
- The executor's runtime binding table (§7.2) becomes a **binding history ring** rather than a single-slot table. Small change; big consistency win.
- The `Backend` interface (Phase 4+) includes the two hooks above from day one.

The metamodel (§4.1) does not change: `FrameProvenance` is runtime metadata attached to stream elements, not an IR-level type or node kind.

### 7.10 Feedback loops via `store_variable`

Some pipelines are inherently closed-loop: raw channel data feeds an on-line SoS estimator whose output updates the TX delays used for future acquisitions; an adaptive TGC estimator watches image intensity and adjusts the gain curve; a rate controller watches the buffer fill level and tunes the PRF. The IR must express these without violating either SSA (no reassignment) or the DAG discipline (no cyclic def-use edges).

**Approach.** Close the loop through the runtime **binding table**, not through an SSA edge. The mechanism is one new op:

```
arrus.store_variable(name: string, value: Value)   # no result; side-effecting sink
```

At runtime, when the containing region fires and `store_variable` executes, it writes `value` into the runtime binding for the named Variable. Consumers of that Variable see the new value on their next fire. The IR stays acyclic; SSA discipline is preserved because `store_variable` produces no new SSA value.

**Example: adaptive SoS.**

```python
sos = arrus.Variable("sos", initial=1540.0)                    # reads from binding table

# Acquisition path (Region A, Us4R)
delays = arrus.compute_delays(sos, geometry, aperture, focal_points)
tx     = arrus.compose_tx(aperture, pulse, delays)
raw    = arrus.acquire(arrus.tx_rx(tx, rx, trigger=clk))

# Estimator path (Region B, GPU, per-N-frame batch)
batch      = arrus.window(raw, size=N)                          # gather slow-time samples
sos_est    = arrus.estimate_sos(batch)                          # Stream[f32], one value per batch

# Feedback: close the loop through the binding table
arrus.store_variable("sos", sos_est)                            # side-effecting sink
```

**Why this is not a cycle in the IR.** The SSA graph reads:
- `sos` (defined by the `Variable` op) → operand of `compute_delays`.
- `raw` → `batch` → `sos_est` → operand of `store_variable`.
- `store_variable` has no results; nothing consumes anything from it.

There is no edge from `store_variable` back to the `Variable` op. The link between them is a *runtime* one (they share a binding-table entry), not a *compile-time* SSA one.

**Timing semantics.**
- Reads happen at operand-read time — the beginning of the reader's fire, per the general firing model.
- Writes happen at op-execution time — when the region containing `store_variable` fires.
- The Variable's initial value (from the `Variable` op's `initial_type`/`initial_data`) is used before any `store_variable` has executed.
- If reader and writer live in different trigger regions running at different rates (typical: acquisition at 5 kHz, estimator at 100 Hz), the reader sees the most recent write on its next fire. Latest-wins.

**Reuse of the existing update pipeline.** A `store_variable` firing invokes the same warm-path partial-rewrite pipeline as `session.set_variable` from outside: the precomputed dirty set for the Variable name is looked up; affected host-side ops re-execute; `Backend.update(handle, changed_fields)` writes only the changed field(s) to shadow device memory; the barrier flips buffers between TX/RXs. Acquisition never pauses. The externally-triggered path and the in-graph-triggered path converge on the same code.

**What this rules out.** Pipelines that need *cycle-by-cycle* pairing (e.g., an IIR filter with recursive coefficients where each output must be paired with the previous input) — Variable-mediated feedback gives latest-wins semantics, not paired-cycle semantics. If a future need arises for tight per-cycle feedback (autoregressive filters over slow-time, etc.), we add a dedicated `delay`/register op then. For every representative use case discussed so far (adaptive SoS, adaptive TGC, buffer-fill-driven PRF control), latest-wins via `store_variable` fits.

**Verifier rules.** New sink-side constraints (added to §4.1 metamodel + verifier):
- `store_variable`'s `name` attribute must match some `Variable(name, ...)` op declared elsewhere in the graph.
- The type of `store_variable`'s `value` operand must match the target Variable's declared type. (Strict — mismatched shapes/dtypes are rejected at graph-construction time, same rule as `set_variable`.)
- A Variable may have multiple `store_variable` writers but must have exactly one `Variable(name)` declaration.

**MLIR mapping.** `store_variable` corresponds exactly to `ml_program.global_store` in MLIR: a side-effecting op that writes a value into a named global cell. The `Variable` op corresponds to `ml_program.global` + `ml_program.global_load`. No adjustment needed at migration time.

## 8. Serialization (Protocol Buffers)

Sketch of `arrus/core/io/proto/ndgraph.proto`:

```proto
syntax = "proto3";
package arrus.graph;

message NdGraph {
  uint32 abi_version = 1;
  repeated Value values = 2;
  repeated Op ops = 3;
  repeated uint32 inputs = 4;
  repeated uint32 outputs = 5;
  repeated Variable variables = 6;
  string producer = 7;
}

message NdArrayType {
  repeated int64 shape = 1;              // -1 for unknown dim
  DType dtype = 2;
  string device = 3;                     // "ANY", "CPU", "GPU:0", "Us4R:0", ...
}

enum DType {
  DTYPE_UNKNOWN = 0; DTYPE_BOOL = 1;
  DTYPE_INT16 = 2;   DTYPE_UINT8 = 3;
  DTYPE_INT32 = 4;   DTYPE_INT64 = 5;
  DTYPE_FLOAT32 = 6; DTYPE_FLOAT64 = 7;
  DTYPE_COMPLEX64 = 8; DTYPE_COMPLEX128 = 9;
}

message Value    { NdArrayType type = 1; string name = 2; }
message Op       { string name = 1; repeated uint32 operands = 2;
                   repeated uint32 results = 3; map<string, Attribute> attrs = 4; }
message Variable { uint32 value = 1; string name = 2; NdArrayData initial = 3; }

message Attribute { oneof v { int64 i = 1; double f = 2; string s = 3;
                              bool b = 4; NdArrayData a = 5; AttrList list = 6; } }
message NdArrayData { NdArrayType type = 1; bytes raw = 2; }
message AttrList    { repeated Attribute items = 1; }

message GraphHistory {
  uint32 abi_version = 1;
  NdGraph base = 2;
  repeated Revision revisions = 3;
}
message Revision {
  string id = 1;
  string parent_id = 2;
  oneof body { NdGraph snapshot = 3; EditSet delta = 4; }
  string producer = 5;
  int64 timestamp_utc_ns = 6;
}
message EditSet { map<string, Attribute> edits = 1; }
```

- Files carry a magic header + `abi_version`. Loader refuses unknown-major files with a clear diagnostic (same policy as ARRUS adapter ABI).
- The proto ABI is a promise; changes within a major version must be additive.

## 9. Interop with JAX and PyTorch

Both adapters live behind optional imports (guarded by `is_package_available`, reused from `arrus/utils/imaging.py:37`).

- **`arrus.numpy.interop.jax`**
  - `from_jax(jaxpr_or_traced_fn, *example_inputs) -> arrus.Graph`
  - `to_jax(graph) -> callable` (jit-able)
- **`arrus.numpy.interop.torch`**
  - `from_torch_fx(gm: torch.fx.GraphModule) -> arrus.Graph`
  - `to_torch_fx(graph) -> torch.fx.GraphModule`

Each adapter maintains an explicit op-mapping table. Unmapped ops raise `UnsupportedOpError` naming the missing entry. Interop is best-effort within the v1 processing op set.

## 10. Bridge to existing `arrus.utils.imaging.Pipeline`

Two additions, no removals — existing user code keeps working:

1. `Operation.to_ndgraph(input_types: List[NdArrayType]) -> arrus.Graph`. Default raises `NotImplementedError`. Ops migrate opportunistically by overriding this.
2. `NdGraphOperation(Operation)` wraps a saved `.ndgraph` so an existing `Pipeline` can execute a graph as one of its steps.

Pilot: migrate one op end-to-end (e.g., a bandpass filter or decimator) as the first bridge validation.

## 11. Python and C++ frontend

- **Python:** `import arrus.numpy as anp`. `anp.zeros`, `anp.fft.fft`, arithmetic operators on `NdArray`, `with arrus.Graph() as g:` context manager. Uses default graph if no context is active.
- **C++:** `arrus::numpy::NdArray` with operator overloads and factory functions. Same default-graph stack semantics; C++ users can use `arrus::Graph` as an RAII scope.
- **Binding layer:** **pybind11**, in a new extension module `arrus._ndgraph`. Existing SWIG-generated `arrus.core` stays as-is; new bindings are strictly additive.

## 12. Compiled-graph lifecycle

`session.compile(graph, target=...)` is the single "prepare" step. It runs placement resolution, shape/type inference, constant folding, DCE, cross-backend transfer planning, per-Variable dirty-set precomputation (§7.2), and per-backend `upload(op, ...)` calls (which upload descriptors, allocate buffers, JIT kernels). Returns a `CompiledGraph` object.

Ops are not equipped with a `prepare()` method of their own — shape/type inference happens eagerly at graph-construction time, and resource allocation happens inside `compile()`. Op authors provide:
- `signature` + `infer_types()` — used during graph construction.
- (Per backend) `upload`, `dispatch`, `update`, and optionally `start`/`stop` — used during compile and at runtime.

### 12.1 `CompiledGraph` API

Uniform API across all trigger-source kinds; the user calls whichever methods make sense for the graph they built.

```python
compiled = session.compile(g, target={"acquisition": "Us4R:0", "processing": "GPU:0"})

# For autonomously-triggered regions (arrus.trigger.internal / external):
compiled.start()          # arm all autonomous trigger sources; hardware begins firing
compiled.stop()           # disarm
# (fine-grained: compiled.start(source_name) / compiled.stop(source_name))

# For software-triggered regions (arrus.trigger.software):
compiled.trigger("frame") # fires one execution of the region rooted at that source

# Convenience for pure-processing graphs (single software root):
outputs = compiled.run(inputs)   # fires the root region once, returns outputs

# Runtime updates:
compiled.set_variable("sos", 1500.0)        # Variable rebind (see §7)
compiled.set_variables({"sos": 1500, "tgc": [...]})

# Release resources:
compiled.close()
```

Ops in autonomously-triggered regions fire without host involvement between `start()` and `stop()` — for `Us4RBackend` acquisition, this is the whole point (constant PRF, no host overhead per TX/RX cycle). Processing regions downstream of an autonomous acquisition region are triggered implicitly by buffer arrival: they fire when new data is available. **The only host actions during a live acquisition session are `set_variable` calls and, at the coarse end, `stop`/`close`.**

### 12.2 Recompile on `copy_with`

Structural edits invalidate the compiled graph:

```python
new_g        = g.copy_with(edits={"depth": 60})
new_compiled = session.compile(new_g, target={...})   # new upload, new descriptor
compiled.stop(); compiled.close()                     # release old device resources
new_compiled.start()
```

Whether the transition can be **gap-free** (double-buffered device state so acquisition never pauses) depends on the backend and the specific edit; `Us4RBackend` already exposes an in-place sub-sequence swap (see `docs/td/ultrasound_interface.md`, `setSubsequences`) that some `copy_with` edits could be lowered to. Not all structural edits qualify (e.g. changing sample count reshapes buffers). The backend advertises which edit kinds it can apply gap-free.

## 13. Future MLIR migration — what to preserve

The IR is deliberately shaped so a later port to MLIR (custom `arrus` dialect) is mechanical. Preserve the following disciplines throughout v1:

- SSA-style values with explicit def-use.
- Immutable ops; rewrites produce new ops and `replace_all_uses_with` re-points consumers.
- Typed values (`NdArrayType` on every `Value`).
- Plain-data attributes only.
- Op registry with a signature and verifier per op.

**Decisions to lock in v1** (semantics we commit to now so they aren't debated at migration time):

- **`Variable` = `ml_program.global`.** A named module-scope mutable cell that ops read at execution time (not a function-argument-rebind model). `set_variable` mutates the cell; readers see the new value on next fire. `Variable` result values are SSA-immutable; only the *memory referenced by them* mutates.
- **`Stream[T]` maps to a custom MLIR type.** SSA discipline is preserved (values are immutable; the *sequence of elements observed at runtime* is a separate concept, like function arguments).
- **Trigger source ops map to root ops in the `arrus` dialect.** Autonomous trigger firing is analogous to a target-specific runtime — the compiled program declares which ops are trigger sources; the backend/runtime schedules them. `dispatch` and `update` map to standard MLIR ops; `start`/`stop` map to runtime-side hooks not visible in the IR.
- **Region partitioning is emergent, not IR-level.** Trigger regions are computed from graph reachability; they are not modeled as explicit `mlir::Region` structures. Optional: for finer control, later versions may add region ops.

**Known compromise:** persistent-data-structure sharing between graph revisions does not survive verbatim to MLIR (MLIR ops are mutable-in-memory and owned by their parent block). At MLIR conversion time, each revision materializes as a distinct module (or the shared sub-DAG becomes a `func.func` called by multiple modules). This is a memory-efficiency loss, not a semantic one.

## 14. Implementation phases

1. Rename `NdArray` → `NdStorage` across C++, SWIG, Python (both types stay in `arrus::framework`; the freed `NdArray` name is claimed by the lazy graph-node type in Phase 2). Publish this design doc.
2. Core IR (C++): `NdArrayType`, `DType`, `DeviceRef`, `Value`, `Op`, `Variable`, `TriggerSource`, `store_variable` sink, immutable persistent `Graph`, `Rewriter` (`replace_all_uses_with` + DCE + `copy_with`), **multi-dialect** `OpRegistry` (`dialect_name → {op_name → signature}` — supports the two-level model of §6.1 from day one), verifier. C++ unit tests.
3. Proto + serializer: `ndgraph.proto`, `NdGraphSerializer` (graph + `GraphHistory` round-trip), tests.
4. `CpuBackend` + first ops (arithmetic, shape, reductions, indexing).
5. Executor + shape/type-inference pass + constant folding.
6. pybind11 extension `arrus._ndgraph`.
7. Python frontend `arrus.numpy` (NdArray class, factories, ops, default-graph stack, Variable, GraphHistory).
8. `CudaBackend` (cupy). Cross-backend equivalence tests.
9. FFT + complex + ultrasound ops on both backends.
10. Bridge: `Operation.to_ndgraph()` + `NdGraphOperation`. Migrate one pilot op.
11. Interop adapters (`interop.jax`, `interop.torch`).
12. User-facing docs + tutorial.
13. **v1.1** — acquisition op family + `Us4RBackend`. Pilot: a B-mode scan expressed as one graph mixing acquisition and processing ops with a shared `Variable("sos")`.

## 15. Verification

- Per-op per-backend unit tests (`unittest`, `*_test.py`).
- Cross-backend equivalence: CPU vs CUDA numeric parity within tolerance.
- Serialization roundtrip: graph and `GraphHistory` with multiple revisions.
- Variable update: multi-consumer coherence.
- `copy_with`: shape propagation correct, unaffected sub-DAG shared by identity.
- JAX / PyTorch interop roundtrip over the v1 op subset.
- Bridge regression: existing `arrus/utils/tests/processing_test.py` and `arrus/utils/tests/imaging/*` still pass; pilot op's `to_ndgraph()` path is numerically equal to the original `Operation`.
- (v1.1) Unified acquisition + processing graph: `Us4RBackend` + `CudaBackend` cooperate over shared `Variable("sos")`; updating the variable affects both without a graph rewrite.

## 16. Figures

Illustrative TikZ figures live in [`figures/`](figures/). Each `.tex` is a standalone document; build with `pdflatex <name>.tex`.

- [`figures/ndarray_fig1_highlevel.tex`](figures/ndarray_fig1_highlevel.tex) — High-level graph: TX/RX → Beamforming → { B-mode, Color Doppler }. Colors indicate device placement.
- [`figures/ndarray_fig2_midlevel.tex`](figures/ndarray_fig2_midlevel.tex) — Op-by-op view of `arrus.utils.imaging.get_bmode_imaging` (LinSequence variant) with a parallel Color Doppler branch forking off the IQ data.
- [`figures/ndarray_fig3_trigger_buffers.tex`](figures/ndarray_fig3_trigger_buffers.tex) — Same high-level graph augmented with an explicit trigger-source node, stream buffers between processing stages, and a `Variable("sos")` consumed by both TX/RX (delays) and Beamforming.
- [`figures/ndarray_common.tex`](figures/ndarray_common.tex) — Shared TikZ preamble (color palette + node styles).

**Metamodels** (see §4.1). ARRUS carries three layered metamodels; two are declared as YAML here, one is planned as an op-catalog YAML (see §5).

- **L1 — Graph of operations on devices** (device-agnostic; what constitutes a well-formed op graph):
  - [`figures/ndarray_metamodel.yaml`](figures/ndarray_metamodel.yaml) — source of truth: node kinds, edge kinds, node/edge attributes, cardinalities, named constraints.
  - [`figures/ndarray_metamodel.png`](figures/ndarray_metamodel.png) / [`.svg`](figures/ndarray_metamodel.svg) — auto-generated UML class diagram (attributes rendered UML-style inside each class box; associations labelled with name and endpoint cardinality).
- **L3 — ARRUS ultrasound domain model** (reverse-engineered from `api/python/arrus/devices/*.py`; what an ultrasound device / probe / probe-model *is*):
  - [`figures/arrus_domain_model.yaml`](figures/arrus_domain_model.yaml) — classes (`Us4R`, `Us4OEM`, `Probe`, `ProbeModel`, `Lens`, `MatchingLayer`, `Backplane`, `HVPSMeasurement`, `CPU`, `GPU`, `GpuSettings`, `DeviceId`, `ProbeModelId`), their attributes with types + units, and relations (`has_us4oem`, `has_probe`, `is_instance_of`, `has_lens`, ...).
  - [`figures/arrus_domain_model.png`](figures/arrus_domain_model.png) / [`.svg`](figures/arrus_domain_model.svg) — auto-generated UML class diagram.
- **Shared generator** — [`figures/generate_metamodel_diagram.py`](figures/generate_metamodel_diagram.py). Takes a YAML path argument (`python3 generate_metamodel_diagram.py [YAML]`); consumes L1 by default, or `arrus_domain_model.yaml` for L3. Emits DOT, PNG, SVG, plus a Markdown auto-summary.
- **L1 ↔ L3 link.** The L1 `Device` node kind carries a `model_ref` attribute pointing at an L3 concrete instance (e.g., `Us4R`). L1 validates op-graph shape; L3 validates ultrasound configuration; they compose without overlap.
- [`figures/qualified_id_demo.py`](figures/qualified_id_demo.py) — proof-of-shape for derived path-like IDs over composite Ops and Devices (§4.1.1).

**Animations** — illustrate the two runtime update mechanisms of §7 on a sequence-expanded variant of fig 3 (four explicit `tx_rx_*` ops in Region A, connected by dotted precedence edges). Both are regenerated by [`figures/animate.py`](figures/animate.py) (requires `pdflatex`, `pdftoppm`, ImageMagick `convert`; `gifsicle` optional for size reduction).

- [`figures/ndarray_anim1_variable_rebind.gif`](figures/ndarray_anim1_variable_rebind.gif) — `session.set_variable("sos", 1500.0)` propagating as an impulse from the Variable node along the def-use edges to the TX/RX delays and to the beamformer; illustrates the BFS-precomputed dirty set and the partial device-memory rewrite (§7.2, §7.3). Acquisition never pauses.
- [`figures/ndarray_anim2_structural_rewrite.gif`](figures/ndarray_anim2_structural_rewrite.gif) — `session.compile(g.copy_with(edits={"n_tx_rx": 2}), ...)` reducing the sequence from four TX/RX ops to two; illustrates the structural-rewrite path (§7.4): revision produced, old `CompiledGraph` stopped, new (shorter) descriptor re-uploaded, acquisition resumed on the new revision.
