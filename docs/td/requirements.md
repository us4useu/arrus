ARRUS is a Software-Defined Ultrasound (SDU) framework:

- ARRUS provides a generic `Ultrasound` device interface, exposed as a stable C ABI. This interface is implemented by external adapter components loaded at runtime. In particular, US4R-HAL ships an ARRUS-adapter (as a separate library, e.g., `.dll`/`.so`) that implements the `Ultrasound` interface on top of the US4R-HAL native API. File-based devices (e.g., datasets) and, in the future, software-based ultrasound simulators are provided in the same way.
    - The `Ultrasound` interface is versioned independently of the ARRUS release version (`arrus-abi X.Y`). Within a major ABI version, ARRUS preserves backward compatibility of the interface: existing symbols, types, and semantics must not be changed or removed; only additive changes are permitted.
    - Each adapter declares two things:
        - **Loader compatibility range** — the range of `arrus-abi` versions it is safe to load with (e.g., `arrus-abi >= 1.0, < 2.0`).
        - **Built-against version** — the specific `arrus-abi` version the adapter was compiled against (e.g., `arrus-abi 1.0`), which bounds the actual method surface it implements.

      ARRUS makes no forward-compatibility promise beyond preserving the C ABI within a major version; the burden of declaring compatibility lies with the adapter, not with ARRUS.
    - ARRUS refuses to load an adapter whose loader range does not include the running `arrus-abi` version, and reports a clear diagnostic.
    - Additive changes to the *core* interface within a major ABI version are gated by ARRUS, not by the caller. When a client invokes a method that was introduced in an ABI version newer than the adapter's built-against version, ARRUS raises a `NotSupported` error that identifies the adapter, its built-against version, and the required version. Feature detection (e.g., `dev.supports("foo")`) is available but not required — callers may simply call and handle the error. Adapters do not need to be republished for each ARRUS release; callers that only use methods present in the adapter's built-against ABI work unchanged. This mechanism covers *evolution of the standard interface*. *Optional device-specific* functionality (e.g., temperature reading, HV control) is handled separately via the capability mechanism described in the last bullet of this list.
    - In the context of system testing (V&V), the specific (ARRUS, adapter) version combinations exercised during testing must be explicitly documented. Users are informed which combinations have been tested by us; other combinations may work by virtue of the ABI contract but are not guaranteed.
- ARRUS provides a framework for defining ultrasound acquisition and processing pipelines as **schemes** — graphs of typed operations whose execution is placed on concrete devices (sensors, hardware accelerators, or CPU/GPU backends). The graph representation is built on top of the `NdArray` operation graph described in the section below.
- ARRUS includes a library of standard ultrasound signal-processing algorithms. In particular, it provides implementations of raw data preprocessing (filtering, demodulation, etc.), B-mode image reconstruction (for linear, convex, and phased, matrix-array probes), and Color Doppler imaging (for linear probes). ARRUS 1.0.0 implements this signal-processing library for both CPUs and NVIDIA GPUs.
- ARRUS provides the Ultrasound Graph Format (UGF), a data storage format intended to unify the formats used for ultrasound datasets.
- Hardware-specific functionality is not part of the generic `Ultrasound` interface. Instead, each adapter exposes named **capabilities** that a caller can query at runtime (e.g., `dev.getCapability("us4useu.us4r.v1")`). The concrete interface returned by such a query is defined by the driver SDK — ARRUS itself has no knowledge of it and only brokers the opaque handle between the adapter and the caller. Requesting a device-specific handle like `"Us4R:0"` is equivalent to requesting the generic `"Ultrasound:0"` handle *plus* the driver-specific capability the SDK publishes (for example, to read the temperature of a selected Us4OEM device).

## NdArray

ARRUS provides an `NdArray` abstraction that represents multi-dimensional arrays and computations over them. The requirements are:

1. **NumPy-like interface.** The `NdArray` interface exposes an n-dimensional array API compatible in spirit with NumPy and CuPy: factory functions (`zeros`, `ones`, `arange`, `array`, …), arithmetic and elementwise operators, reductions (`sum`, `mean`, …), shape manipulation (`reshape`, `transpose`, broadcasting), indexing/slicing, and dtypes. The intent is that code written against `arrus.numpy` reads like code written against `numpy`/`cupy`.

2. **Lazy evaluation via an operation graph.** Operations on `NdArray` instances are not executed eagerly. Instead, each operation appends a node to an underlying graph of operations. The graph describes computation on an abstract device — placement on a concrete device (CPU, CUDA GPU, custom accelerator) is a separate concern resolved at compile/execute time.

3. **Explicit graphs and default graph; on-disk storage.** A default graph is used implicitly when the user creates `NdArray`s without a surrounding graph context. Users can also create an explicit graph and bind inputs/outputs/variables to it using a context manager:
   ```python
   import arrus.numpy as anp
   import arrus

   with arrus.Graph() as g:
       x = anp.zeros(3)
       y = x + 1
       g.mark_output(y)

   g.save("pipeline.ugraph")
   g2 = arrus.Graph.load("pipeline.ugraph")
   ```
   Graphs are serializable to and loadable from disk in a stable, versioned format.

4. **Multiple backends, extensible.** ARRUS ships reference implementations of graph operations for CPU and NVIDIA CUDA GPUs. The backend interface is defined so that additional backends — other GPU vendors (e.g., ROCm, oneAPI), custom FPGAs, or dedicated accelerators — can be added out of tree without modifying the core.

5. **Interoperability with JAX and PyTorch graphs.** ARRUS supports converting graphs to and from the graph representations used by JAX (jaxpr / StableHLO) and PyTorch (`torch.fx` / TorchScript / ExportedProgram), so that pipelines can be authored in either ecosystem and executed by ARRUS, and vice versa.

An ARRUS **Scheme** — an ultrasound acquisition + processing pipeline — is expressed as an `arrus.Graph` whose nodes include ultrasound-specific operations (e.g., `Tx`, `Rx`, `TxRx`, `DDC`, `Beamform`, `LogCompress`). Each node carries an `isPlaced` relation to a sensor (e.g., `Probe:0`) or a device (e.g., `Us4R:0`, `GPU:0`); the backend responsible for executing the node is determined from that placement. See `scheme_graph.drawio` for a concrete example.
