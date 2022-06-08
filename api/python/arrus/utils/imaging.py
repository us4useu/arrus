import numpy as np
import math
import scipy
import scipy.signal as signal
import scipy.ndimage
import arrus.metadata
import arrus.devices.device
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.kernels.imaging
import arrus.utils.us4r
import arrus.ops.imaging
import arrus.ops.us4r
import queue
import dataclasses
import threading
from collections import deque
from collections.abc import Iterable
from pathlib import Path
import os
import importlib.util


def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None


if is_package_available("cupy"):
    import cupy
    import re
    if not re.match("^\\d+\\.\\d+\\.\\d+$", cupy.__version__):
        raise ValueError(f"Unrecognized pattern "
                         f"of the cupy version: {cupy.__version__}")
    if tuple(int(v) for v in cupy.__version__.split(".")) < (9, 0, 0):
        raise Exception(f"The version of cupy module is too low. "
                        f"Use version ''9.0.0'' or higher.")
else:
    print("Cupy package is not available, some of the arrus.utils.imaging "
          "operators may not be available.")


def get_extent(x_grid, z_grid):
    """
    A simple utility tool to get output image extents:

    The output format is compatible with matplotlib:
    [ox_min, ox_max, oz_max, oz_min]
    """
    return np.array([np.min(x_grid), np.max(x_grid),
                     np.max(z_grid), np.min(z_grid)])


def get_bmode_imaging(sequence, grid, placement="/GPU:0",
                      decimation_factor=4, decimation_cic_order=2):
    """
    Returns a standard B-mode imaging pipeline.

    :param sequence: TX/RX sequence for which the processing has to be performed
    :param grid: output grid, a pair (x_grid, z_grid), where x_grid/z_grid
      are the OX/OZ coordinates of consecutive grid points
    :param placement: device on which the processing should be performed
    :param decimation_factor: decimation factor to apply
    :param decimation_cic_order: decimation CIC filter order
    :return: B-mode imaging pipeline
    """
    x_grid, z_grid = grid
    if isinstance(sequence, arrus.ops.imaging.LinSequence):
        # Classical beamforming.
        return Pipeline(
            steps=(
                # Channel data pre-processing.
                RemapToLogicalOrder(),
                Transpose(axes=(0, 1, 3, 2)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=decimation_factor,
                           cic_order=decimation_cic_order),
                # # Data beamforming.
                RxBeamforming(),
                # # Post-processing to B-mode image.
                EnvelopeDetection(),
                Transpose(axes=(0, 2, 1)),
                ScanConversion(x_grid, z_grid),
                Mean(axis=0),
                LogCompression(),
            ),
            placement=placement)
    elif isinstance(sequence, arrus.ops.imaging.PwiSequence) \
      or isinstance(sequence, arrus.ops.imaging.StaSequence):
        # Synthetic aperture imaging.
        return Pipeline(
            steps=(
                # Channel data pre-processing.
                RemapToLogicalOrder(),
                Transpose(axes=(0, 1, 3, 2)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=decimation_factor,
                           cic_order=decimation_cic_order),
                # Data beamforming.
                ReconstructLri(x_grid=x_grid, z_grid=z_grid),
                # IQ compounding
                Mean(axis=1),  # Along tx axis.
                # Post-processing to B-mode image.
                EnvelopeDetection(),
                # Envelope compounding
                Mean(axis=0),
                Transpose(),
                LogCompression()
            ),
            placement=placement)
    else:
        raise ValueError(f"Unrecognized imaging TX/RX sequence: {sequence}")


class BufferElement:
    """
    This class represents a single element of data buffer.
    Acquiring the element when it's not released will end up with
    an "buffer override" exception.
    """

    def __init__(self, pos, data):
        self.pos = pos
        self.data = data
        self.size = self.data.nbytes
        self.occupied = False
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            if self.occupied:
                raise ValueError("GPU buffer override")
            self.occupied = True

    def release(self):
        with self._lock:
            self.occupied = False


class BufferElementLockBased:
    """
    This class represents a single element of data buffer.
    Acquiring the element when it's not released will block the caller
    until the element is released.
    """
    def __init__(self, pos, data):
        self.pos = pos
        self.data = data
        self.size = self.data.nbytes
        self._semaphore = threading.Semaphore()
        # The acquired semaphore means, that the buffer element is still
        # in the use and cannot be filled with new data coming from producer.

    def acquire(self):
        self._semaphore.acquire()

    def release(self):
        self._semaphore.release()


class Buffer:

    def __init__(self, n_elements, shape, dtype, math_pkg, type="locked"):
        element_type = None
        if type == "locked":
            element_type = BufferElementLockBased
        elif type == "async":
            element_type = BufferElement
        else:
            raise ValueError(f"Unrecognized buffer type: {type}")
        self.input_array = [math_pkg.zeros(shape, dtype=dtype) for _ in range(n_elements)]
        self.elements = [element_type(i, data) for i, data in enumerate(self.input_array)]
        self.n_elements = n_elements

    def acquire(self, pos):
        element = self.elements[pos]
        element.acquire()
        return element

    def release(self, pos):
        self.elements[pos].release()


class ProcessingRunner:
    """
    Currently, the input buffer should be located in CPU device,
    output buffer should be located on GPU.
    """

    def __init__(self, input_buffer, const_metadata, processing):
        import cupy as cp
        # Initialize pipeline.
        self.cp = cp
        self.input_buffer = self.__register_buffer(input_buffer)
        default_buffer = ProcessingBuffer(size=2, type="locked")

        in_buffer_spec = processing.input_buffer
        out_buffer_spec = processing.output_buffer
        in_buffer_spec = in_buffer_spec if in_buffer_spec is not None else default_buffer
        out_buffer_spec = out_buffer_spec if out_buffer_spec is not None else default_buffer

        self.gpu_buffer = Buffer(n_elements=in_buffer_spec.size,
                                 shape=const_metadata.input_shape,
                                 dtype=const_metadata.dtype, math_pkg=cp,
                                 type=in_buffer_spec.type)
        self.pipeline = processing.pipeline
        self.data_stream = cp.cuda.Stream(non_blocking=True)
        self.processing_stream = cp.cuda.Stream(non_blocking=True)
        self.out_metadata = processing.pipeline.prepare(const_metadata)
        self.out_buffers = [Buffer(n_elements=out_buffer_spec.size, shape=m.input_shape,
                                  dtype=m.dtype, math_pkg=np,
                                  type=out_buffer_spec.type)
                           for m in self.out_metadata]
        # Wait for all the initialization done in by the Pipeline.
        cp.cuda.Stream.null.synchronize()
        self.out_buffers = self.__register_buffer(self.out_buffers)
        if not isinstance(self.out_buffers, Iterable):
            self.out_buffers = (self.out_buffers, )
        self._process_lock = threading.Lock()
        if processing.callback is not None:
            self.user_out_buffer = None
            self.callback = processing.callback
        else:
            self.user_out_buffer = queue.Queue(maxsize=1)
            self.callback = self.default_callback
        self._gpu_i = 0
        self._out_i = [0]*len(self.out_buffers)
        self.i = 0
        # Metadata extraction.
        self.is_extract_metadata = processing.extract_metadata
        if self.is_extract_metadata:
            self.metadata_extractor = ExtractMetadata()
            self.metadata_extractor.prepare(const_metadata)
        self.input_buffer.append_on_new_data_callback(self.process)
        if processing.on_buffer_overflow_callback is not None:
            self.input_buffer.append_on_buffer_overflow_callback(
                processing.on_buffer_overflow_callback)

    @property
    def outputs(self):
        const_metadata = None
        if len(self.out_metadata) == 1:
            const_metadata = self.out_metadata[0]
        else:
            const_metadata = self.out_metadata
        if self.user_out_buffer is not None:
            return self.user_out_buffer, const_metadata
        else:
            return const_metadata

    def default_callback(self, elements):
        try:
            user_elements = [None]*len(elements)
            for i, element in enumerate(elements):
                user_elements[i] = element.data.copy()
                element.release()
            try:
                self.user_out_buffer.put_nowait(user_elements)
            except queue.Full:
                pass
        except Exception as e:
            print(f"Exception: {type(e)}")
        except:
            print("Unknown exception")

    def process(self, input_element):
        with self._process_lock:
            if self.is_extract_metadata:
                metadata = self.metadata_extractor.process(input_element.data)
            gpu_element = self.gpu_buffer.acquire(self._gpu_i)
            gpu_array = gpu_element.data
            self._gpu_i = (self._gpu_i+1) % self.gpu_buffer.n_elements
            gpu_array.set(input_element.data, stream=self.data_stream)
            self.data_stream.launch_host_func(self.__release, input_element)

            gpu_data_ready_event = self.data_stream.record()
            self.processing_stream.wait_event(gpu_data_ready_event)

            out_elements = []
            with self.processing_stream:
                results = self.pipeline(gpu_array)
                # Write each result gpu array to given output array
                for i, (result, out_buffer) in enumerate(zip(results, self.out_buffers)):
                    out_i = self._out_i[i]
                    out_element = out_buffer.elements[out_i]
                    self._out_i[i] = (out_i+1) % out_buffer.n_elements
                    self.processing_stream.launch_host_func(
                        lambda element: element.acquire(), out_element)
                    result.get(self.processing_stream, out=out_element.data)
                    out_elements.append(out_element)
            if self.is_extract_metadata:
                out_elements.insert(0, metadata)
            self.processing_stream.launch_host_func(self.__release, gpu_element)
            self.processing_stream.launch_host_func(self.callback, out_elements)

    def stop(self):
        # cleanup
        self.__unregister_buffer(self.input_buffer)
        self.__unregister_buffer(self.out_buffers)

    def sync(self):
        self.data_stream.synchronize()
        self.processing_stream.synchronize()

    def __release(self, element):
        element.release()

    def __register_buffer(self, buffers):
        import cupy as cp
        if not isinstance(buffers, Iterable):
            buffers = (buffers, )
        for buffer in buffers:
            for element in buffer.elements:
                cp.cuda.runtime.hostRegister(element.data.ctypes.data,
                                             element.size, 1)
        if len(buffers) == 1:
            buffers = next(iter(buffers))
        return buffers

    def __unregister_buffer(self, buffers):
        import cupy as cp
        if not isinstance(buffers, Iterable):
            buffers = (buffers, )
        for buffer in buffers:
            for element in buffer.elements:
                cp.cuda.runtime.hostUnregister(element.data.ctypes.data)


class Operation:
    """
    An operation to perform in the imaging pipeline -- one data processing
    stage.
    """

    def prepare(self, const_metadata):
        """
        Function that will be called when the processing pipeline is prepared.

        :param const_metadata: const metadata describing output from the \
          previous Operation.
        :return: const metadata describing output of this Operation.
        """
        pass

    def process(self, data):
        """
        Function that will be called when new data arrives.

        :param data: input data
        :return: output data
        """
        raise ValueError("Calling abstract method")

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def initialize(self, data):
        """
        Initialization function.

        This function will be called on a cupy initialization stage.

        By default, it runs `_process` function on a test cupy data.

        :return: the processed data.
        """
        return self.process(data)

    def set_pkgs(self, **kwargs):
        """
        Provides to possibility to gather python packages for numerical
        processing and filtering.

        The provided kwargs are:

        - `num_pkg`: numerical package: numpy for CPU, cupy for GPU
        - `filter_pkg`: scipy.ndimage for CPU, cupyx.scipy.ndimage for GPU
        """
        pass


class Output(Operation):
    """
    Output node.

    Adding this node into the pipeline at a specific path will cause
    adding this operator to the Pipeline will cause the output buffer to
    return data from a given processing step.
    """

    def __init__(self):
        self.endpoint = True

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def initialize(self, data):
        return data

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def process(self, data):
        return (data, )


class Pipeline:
    """
    Imaging pipeline.

    Processes given data using a given sequence of steps.
    The processing will be performed on a given device ('placement').
    :param steps: processing steps to run
    :param placement: device on which the processing should take place,
      default: GPU:0
    :param callback: callback to run when output data is ready. By default
    """
    def __init__(self, steps, placement=None):
        self.steps = steps
        self._placement = None
        self._processing_stream = None
        self._input_buffer = None
        if placement is not None:
            self.set_placement(placement)

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        outputs = deque()  # TODO avoid creating deque on each processing step
        for step in self.steps:
            if step.endpoint:
                step_outputs = step.process(data)
                for output in step_outputs:
                    outputs.appendleft(output)
            else:
                data = step.process(data)
        if not self._is_last_endpoint:
            outputs.appendleft(data)
        return outputs

    def __initialize(self, const_metadata):
        input_shape = const_metadata.input_shape
        input_dtype = const_metadata.dtype
        self._input_buffer = self.num_pkg.zeros(
            input_shape, dtype=input_dtype)+1000
        data = self._input_buffer
        for step in self.steps:
            if not isinstance(step, (Pipeline, Output)):
                data = step.initialize(data)

    def prepare(self, const_metadata):
        metadatas = deque()
        current_metadata = const_metadata
        for step in self.steps:
            if isinstance(step, (Pipeline, Output)):
                child_metadatas = step.prepare(current_metadata)
                if not isinstance(child_metadatas, Iterable):
                    child_metadatas = (child_metadatas, )
                for metadata in child_metadatas:
                    metadatas.appendleft(metadata)
                step.endpoint = True
            else:
                current_metadata = step.prepare(current_metadata)
                step.endpoint = False
        # Force cupy to recompile kernels before running the pipeline.
        self.__initialize(const_metadata)
        last_step = self.steps[-1]
        if not isinstance(last_step, (Pipeline, Output)):
            metadatas.appendleft(current_metadata)
            self._is_last_endpoint = False
        else:
            self._is_last_endpoint = True
        return metadatas

    def set_placement(self, device):
        """
        Sets the pipeline to be executed on a particular device.

        :param device: device on which the pipeline should be executed
        """
        device_type = None
        device_ordinal = 0
        if isinstance(device, str):
            # Device id
            device_type, device_ordinal = arrus.devices.device.split_device_id_str(device)
        elif isinstance(device, arrus.devices.device.DeviceId):
            device_type = device.device_type.type
            device_ordinal = device.ordinal
        elif isinstance(device, arrus.devices.device.Device):
            device_type = device.get_device_id().device_type.type
            device_ordinal = device.get_device_id().ordinal

        if device_ordinal != 0:
            raise ValueError("Currently only GPU (or CPU) :0 are supported.")

        self._placement = device_type
        # Initialize steps with a proper library.
        if self._placement == "GPU":
            import cupy as cp
            import cupyx.scipy.ndimage as cupy_scipy_ndimage
            pkgs = dict(num_pkg=cp, filter_pkg=cupy_scipy_ndimage)
            self._processing_stream = cp.cuda.Stream()
        elif self._placement == "CPU":
            import scipy.ndimage
            pkgs = dict(num_pkg=np, filter_pkg=scipy.ndimage)
        else:
            raise ValueError(f"Unsupported device: {device}")
        for step in self.steps:
            if isinstance(step, Pipeline):
                # Make sure the child pipeline has the same placement as parent
                if step._placement != self._placement:
                    raise ValueError("All pipelines should be placed on the "
                                     "same processing device (e.g. GPU:0)")
            else:
                step.set_pkgs(**pkgs)
        self.num_pkg = pkgs['num_pkg']
        self.filter_pkg = pkgs['filter_pkg']


@dataclasses.dataclass(frozen=True)
class ProcessingBuffer:
    size: int
    type: str
    # TODO: placement


class Processing:
    """
    A description of complete data processing run in the arrus.utils.imaging.
    """

    def __init__(self, pipeline, callback=None, extract_metadata=False,
                 input_buffer: ProcessingBuffer=None,
                 output_buffer: ProcessingBuffer=None,
                 on_buffer_overflow_callback=None):
        self.pipeline = pipeline
        self.callback = callback
        self.extract_metadata = extract_metadata
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.on_buffer_overflow_callback = on_buffer_overflow_callback


class Lambda(Operation):
    """
    Custom function to perform on data from a given step.
    """

    def __init__(self, function, prepare_func=None):
        """
        Lambda op constructor.

        :param function: a function with a single input: (cupy or numpy array \
          with the data)
        """
        self.func = function
        self.prepare_func = prepare_func
        pass

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def initialize(self, data):
        return data

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        if self.prepare_func is not None:
            return self.prepare_func(const_metadata)
        else:
            return const_metadata

    def process(self, data):
        return self.func(data)


def _get_unique_pulse(sequence):
    if isinstance(sequence, arrus.ops.imaging.SimpleTxRxSequence):
        return sequence.pulse
    elif isinstance(sequence, arrus.ops.us4r.TxRxSequence):
        pulses = {tx_rx.tx.excitation for tx_rx in sequence.ops}
        if len(pulses) > 1:
            raise ValueError("Each TX/RX should have exactly the same "
                             "definition of transmit pulse.")
        return next(iter(pulses))


class BandpassFilter(Operation):
    """
    Bandpass filtering to apply to signal data.

    A bandwidth [0.5, 1.5]*center_frequency is currently used.

    The filtering is performed along the last axis.

    Currently only FIR filter is available.

    NOTE: consider using Filter op, which provides more possibilities to
    define what kind of filter is used (e.g. by providing filter coefficients).
    """

    def __init__(self, numtaps=7, bounds=(0.5, 1.5), filter_type="butter",
                 num_pkg=None, filter_pkg=None):
        """
        Bandpass filter constructor.

        :param bounds: determines filter's frequency boundaries,
            e.g. setting 0.5 will give a bandpass filter
            [0.5*center_frequency, 1.5*center_frequency].
        """
        self.taps = None
        self.numtaps = numtaps
        self.bound_l, self.bound_r = bounds
        self.filter_type = filter_type
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        l, r = self.bound_l, self.bound_r
        seq = const_metadata.context.sequence
        center_frequency = _get_unique_pulse(const_metadata.context.sequence).center_frequency
        sampling_frequency = const_metadata.data_description.sampling_frequency
        taps, _ = scipy.signal.butter(
                2,
                [l * center_frequency, r * center_frequency],
                btype='bandpass', fs=sampling_frequency)
        self.taps = self.xp.asarray(taps).astype(self.xp.float32)
        return const_metadata

    def process(self, data):
        result = self.filter_pkg.convolve1d(data, self.taps, axis=-1,
                                            mode='constant')
        return result


class FirFilter(Operation):

    def __init__(self, taps, num_pkg=None, filter_pkg=None):
        """
        Bandpass filter constructor.
        :param bounds: determines filter's frequency boundaries,
            e.g. setting 0.5 will give a bandpass filter
            [0.5*center_frequency, 1.5*center_frequency].
        """
        self.taps = taps
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.convolve1d_func = None
        self.dumped = 0

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        if self.xp == np:
            raise ValueError("This operation is NYI for CPU")
        import cupy as cp
        self.taps = cp.asarray(self.taps).astype(cp.float32)
        n_taps = len(self.taps)

        input_shape = const_metadata.input_shape
        n_samples = input_shape[-1]
        total_n_samples = math.prod(input_shape)

        if total_n_samples == 0:
            raise ValueError("Empty array is not supported")

        fir_output_buffer = cp.zeros(const_metadata.input_shape, dtype=cp.float32)
        from arrus.utils.fir import (
            run_fir_int16,
            get_default_grid_block_size_fir_int16,
            get_default_shared_mem_size_fir_int16
        )
        grid_size, block_size = get_default_grid_block_size_fir_int16(
            n_samples,
            total_n_samples)
        shared_memory_size = get_default_shared_mem_size_fir_int16(
            n_samples, n_taps)

        def gpu_convolve1d(data):
            data = cp.ascontiguousarray(data)
            run_fir_int16(
                grid_size, block_size,
                (fir_output_buffer, data, n_samples,
                total_n_samples, self.taps, n_taps),
                shared_memory_size)
            return fir_output_buffer
        self.convolve1d_func = gpu_convolve1d
        return const_metadata.copy(dtype=self.xp.float32)

    def process(self, data):
        return self.convolve1d_func(data)


class Filter(Operation):
    """
    Filter data in one dimension along the last axis.

    The filtering is performed according to the given filter taps.

    Currently only FIR filter is available.
    """

    def __init__(self, taps, num_pkg=None, filter_pkg=None):
        """
        Bandpass filter constructor.

        :param taps: filter feedforward coefficients
        """
        self.taps = taps
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        self.taps = self.xp.asarray(self.taps).astype(self.xp.float32)
        return const_metadata

    def process(self, data):
        result = self.filter_pkg.convolve1d(data, self.taps, axis=-1,
                                            mode='constant')
        return result


class QuadratureDemodulation(Operation):
    """
    Quadrature demodulation (I/Q decomposition).
    """
    def __init__(self, num_pkg=None):
        self.mod_factor = None
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _is_prepared(self):
        return self.mod_factor is not None

    def prepare(self, const_metadata):
        xp = self.xp
        fs = const_metadata.data_description.sampling_frequency
        fc = _get_unique_pulse(const_metadata.context.sequence).center_frequency
        input_shape = const_metadata.input_shape
        n_samples = input_shape[-1]
        if n_samples == 0:
            raise ValueError("Empty array is not accepted.")
        t = (xp.arange(0, n_samples) / fs).reshape(1, 1, -1)
        self.mod_factor = (2 * xp.cos(-2 * xp.pi * fc * t)
                           + 2 * xp.sin(-2 * xp.pi * fc * t) * 1j)
        self.mod_factor = self.mod_factor.astype(xp.complex64)
        return const_metadata.copy(is_iq_data=True, dtype="complex64")

    def process(self, data):
        return self.mod_factor * data


class Decimation(Operation):
    """
    Decimation + CIC (Cascade Integrator-Comb) filter.

    See: https://en.wikipedia.org/wiki/Cascaded_integrator%E2%80%93comb_filter
    """

    def __init__(self, decimation_factor, cic_order, num_pkg=None, impl="legacy"):
        """
        Decimation op constructor.

        :param decimation_factor: decimation factor to apply
        :param cic_order: CIC filter order
        """
        self.decimation_factor = decimation_factor
        self.cic_order = cic_order
        self.xp = num_pkg
        self.impl = impl
        if self.impl == "legacy":
            self._decimate = self._legacy_decimate
        elif self.impl == "fir":
            self._decimate = self._fir_decimate

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg # not used by the GPU implementation (custom kernel for complex input data)

    def prepare(self, const_metadata):
        new_fs = (const_metadata.data_description.sampling_frequency
                  / self.decimation_factor)
        new_signal_description = arrus.metadata.EchoDataDescription(
            sampling_frequency=new_fs, custom=
            const_metadata.data_description.custom)

        input_shape = const_metadata.input_shape
        n_samples = input_shape[-1]
        total_n_samples = math.prod(input_shape)
        output_shape = input_shape[:-1] + (math.ceil(n_samples/self.decimation_factor), )

        # CIC FIR coefficients
        if self.impl == "fir":
            cicFir = self.xp.array([1], dtype=self.xp.float32)
            cicFir1 = self.xp.ones(self.decimation_factor, dtype=self.xp.float32)
            for i in range(self.cic_order):
                cicFir = self.xp.convolve(cicFir, cicFir1, 'full')
            fir_taps = cicFir
            n_fir_taps = len(fir_taps)
            if self.xp == np:
                def _cpu_fir_filter(data):
                    return self.filter_pkg.convolve1d(
                        np.real(data), fir_taps,
                        axis=-1, mode='constant',
                        cval=0, origin=-1) \
                        + self.filter_pkg.convolve1d(np.imag(data),
                                          fir_taps, axis=-1,
                                          mode='constant', cval=0,
                                          origin=-1)*1j
                # CPU
                self._fir_filter = _cpu_fir_filter
            else:
                # GPU
                import cupy as cp
                _fir_output_buffer = cp.zeros(const_metadata.input_shape,
                                                dtype=cp.complex64)
                # Kernel settings
                from arrus.utils.fir import (
                    get_default_grid_block_size,
                    get_default_shared_mem_size,
                    run_fir)
                grid_size, block_size = get_default_grid_block_size(n_samples, total_n_samples)
                shared_memory_size = get_default_shared_mem_size(n_samples, n_fir_taps)

                def _gpu_fir_filter(data):
                    run_fir(grid_size, block_size,
                        (_fir_output_buffer, data, n_samples,
                         total_n_samples, fir_taps, n_fir_taps),
                            shared_memory_size)
                    return _fir_output_buffer

                self._fir_filter = _gpu_fir_filter
        return const_metadata.copy(data_desc=new_signal_description,
                                   input_shape=output_shape)

    def process(self, data):
        return self._decimate(data)

    def _fir_decimate(self, data):
        fir_output = self._fir_filter(data)
        data_out = fir_output[..., 0::self.decimation_factor]
        return data_out

    def _legacy_decimate(self, data):
        data_out = data
        for i in range(self.cic_order):
            data_out = self.xp.cumsum(data_out, axis=-1)
        data_out = data_out[..., 0::self.decimation_factor]
        for i in range(self.cic_order):
            data_out[..., 1:] = self.xp.diff(data_out, axis=-1)
        return data_out


class RxBeamforming(Operation):
    """
    Classical rx beamforming (reconstruct image scanline by scanline).
    This operator implements beamforming for linear scanning (element by element)
    and phased scanning (angle by angle).
    """
    def __init__(self, num_pkg=None):
        # Actual implementation of the operator.
        self._op = None
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata):
        seq = const_metadata.context.sequence
        # Determine scanning type based on the sequence of parameters.
        tx_centers = seq.tx_aperture_center_element
        if tx_centers is None:
            tx_centers = seq.tx_aperture_center
        tx_centers = set(np.atleast_1d(tx_centers))
        tx_angles = set(np.atleast_1d(seq.angles))
        # Phased array scanning:
        # - single TX/RX aperture position
        # - multiple different angles
        if len(tx_centers) == 1 and len(tx_angles) > 1:
            self._op = RxBeamformingPhasedScanning(num_pkg=self.xp)
        # Linear array scanning:
        # - single transmit angle (equal 0)
        # - multiple different aperture positions
        elif len(tx_centers) > 1 and len(tx_angles) == 1:
            self._op = RxBeamformingLin(num_pkg=self.xp)
        # Otherwise: unsupported scanning method (linear/phased)
        else:
            raise ValueError("RX beamformer does not support parameters of "
                             "the provided TX/RX sequence.")
        return self._op.prepare(const_metadata)

    def process(self, data):
        return self._op.process(data)


class RxBeamformingPhasedScanning(Operation):
    """
    Classical beamforming for phase array scanning.
    """

    def __init__(self, num_pkg=None):
        self.num_pkg = num_pkg

    def prepare(self, const_metadata):
        import cupy as cp
        if self.num_pkg != cp:
            raise ValueError("Phased scanning is implemented for GPU only.")
        probe_model = const_metadata.context.device.probe.model
        if probe_model.is_convex_array():
            raise ValueError("Phased array scanning is implemented for "
                             "linear phased arrays only.")

        self._kernel_module = _read_kernel_module("rx_beamforming.cu")
        self._kernel = self._kernel_module.get_function("beamformPhasedArray")

        self.n_seq, self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.output_buffer = cp.zeros((self.n_seq, self.n_tx, self.n_samples), dtype=cp.complex64)

        seq = const_metadata.context.sequence
        self.tx_angles = cp.asarray(seq.angles, dtype=cp.float32)

        device_fs = const_metadata.context.device.sampling_frequency
        acq_fs = (device_fs/seq.downsampling_factor)
        fs = const_metadata.data_description.sampling_frequency
        fc = seq.pulse.center_frequency
        n_periods = seq.pulse.n_periods
        medium = const_metadata.context.medium
        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound
        start_sample, end_sample = seq.rx_sample_range
        initial_delay = - start_sample / acq_fs
        if seq.init_delay == "tx_start":
            burst_factor = n_periods / (2 * fc)
            tx_rx_params = arrus.kernels.imaging.compute_tx_rx_params(
                probe_model, seq, c)
            tx_center_delay = tx_rx_params["tx_center_delay"]
            initial_delay += tx_center_delay + burst_factor
        elif not seq.init_delay == "tx_center":
            raise ValueError(f"Unrecognized init_delay value: {initial_delay}")
        lambd = c / fc
        max_tang = abs(math.tan(
            math.asin(min(1, 2 / 3 * lambd / probe_model.pitch))))

        self.fc = cp.float32(fc)
        self.fs = cp.float32(fs)
        self.c = cp.float32(c)
        # Note: start sample has to be appropriately adjusted for
        # the ACQ sampling frequency.
        self.start_time = cp.float32(start_sample/acq_fs)
        self.init_delay = cp.float32(initial_delay)
        self.max_tang = cp.float32(max_tang)
        sample_block_size = min(self.n_samples, 16)
        scanline_block_size = min(self.n_tx, 16)
        n_seq_block_size = min(self.n_seq, 4)
        self.block_size = (sample_block_size, scanline_block_size, n_seq_block_size)
        self.grid_size = (int((self.n_samples-1)//sample_block_size + 1),
                          int((self.n_tx-1)//scanline_block_size + 1),
                          int((self.n_seq-1)//n_seq_block_size + 1))
        # xElemConst
        # Get aperture origin (for the given aperture center element/aperture center)
        tx_rx_params = arrus.kernels.imaging.preprocess_sequence_parameters(probe_model, seq)
        # There is a single TX and RX aperture center for all TX/RXs
        rx_aperture_center_element = np.array(tx_rx_params["rx_ap_cent"])[0]
        rx_aperture_origin = _get_rx_aperture_origin(
            rx_aperture_center_element, seq.rx_aperture_size)
        rx_aperture_offset = rx_aperture_center_element-rx_aperture_origin
        x_elem = (np.arange(0, self.n_rx)-rx_aperture_offset) * probe_model.pitch
        x_elem = x_elem.astype(np.float32)
        self.x_elem_const = _get_const_memory_array(
            self._kernel_module, "xElemConst", x_elem)
        return const_metadata.copy(input_shape=self.output_buffer.shape)

    def process(self, data):
        data = self.num_pkg.ascontiguousarray(data)
        params = (
            self.output_buffer, data,
            self.n_seq, self.n_tx, self.n_rx, self.n_samples,
            self.tx_angles,
            self.init_delay, self.start_time,
            self.c, self.fs, self.fc, self.max_tang)
        self._kernel(self.grid_size, self.block_size, params)
        return self.output_buffer


class RxBeamformingLin(Operation):

    def __init__(self, num_pkg=None):
        self.delays = None
        self.buffer = None
        self.rx_apodization = None
        self.xp = num_pkg
        self.interp1d_func = None

    def _set_interpolator(self, **kwargs):
        if self.xp is np:
            import scipy.interpolate

            def numpy_interp1d(input, samples, output):
                n_samples = input.shape[-1]
                x = np.arange(0, n_samples)
                interpolator = scipy.interpolate.interp1d(
                    x, input, kind="linear", bounds_error=False,
                    fill_value=0.0)
                interp_values = interpolator(samples)
                n_scanlines, _, n_samples = interp_values.shape
                interp_values = np.reshape(interp_values, (n_scanlines, n_samples))
                output[:] = interp_values

            self.interp1d_func = numpy_interp1d
        else:
            import cupy as cp
            if self.xp != cp:
                raise ValueError(f"Unhandled numerical package: {self.xp}")
            import arrus.utils.interpolate
            self.interp1d_func = arrus.utils.interpolate.interp1d

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        self._set_interpolator()
        context = const_metadata.context
        probe_model = const_metadata.context.device.probe.model
        seq = const_metadata.context.sequence
        raw_seq = const_metadata.context.raw_sequence
        medium = const_metadata.context.medium
        tx_rx_params = arrus.kernels.imaging.preprocess_sequence_parameters(probe_model, seq)
        rx_aperture_center_element = np.array(tx_rx_params["tx_ap_cent"])

        self.n_seq, self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.is_iq = const_metadata.is_iq_data
        if self.is_iq:
            buffer_dtype = self.xp.complex64
        else:
            buffer_dtype = self.xp.float32

        # -- Output buffer
        self.buffer = self.xp.zeros(
            (self.n_seq*self.n_tx, self.n_rx * self.n_samples),
            dtype=buffer_dtype)

        # -- Delays
        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = const_metadata.data_description.sampling_frequency
        fc = seq.pulse.center_frequency
        n_periods = seq.pulse.n_periods
        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound
        tx_angle = 0
        start_sample = seq.rx_sample_range[0]
        rx_aperture_origin = _get_rx_aperture_origin(rx_aperture_center_element, seq.rx_aperture_size)
        # -start_sample compensates the fact, that the data indices always
        # start from 0
        initial_delay = - start_sample / acq_fs
        if seq.init_delay == "tx_start":
            burst_factor = n_periods / (2 * fc)
            tx_rx_params = arrus.kernels.imaging.compute_tx_rx_params(
                probe_model, seq, c)
            tx_center_delay = tx_rx_params["tx_center_delay"]
            initial_delay += tx_center_delay + burst_factor
        elif not seq.init_delay == "tx_center":
            raise ValueError(f"Unrecognized init_delay value: {initial_delay}")
        radial_distance = (
                (start_sample / acq_fs + np.arange(0, self.n_samples) / fs) * c/2
        )
        x_distance = (radial_distance * np.sin(tx_angle)).reshape(1, -1)
        z_distance = radial_distance * np.cos(tx_angle).reshape(1, -1)

        origin_offset = (rx_aperture_origin[0]
                         - (rx_aperture_center_element[0]))
        # New coordinate system: origin: rx aperture center
        element_position = ((np.arange(0, self.n_rx) + origin_offset)
                            * probe_model.pitch)
        element_position = element_position.reshape((self.n_rx, 1))
        if not probe_model.is_convex_array():
            element_angle = np.zeros((self.n_rx, 1))
            element_x = element_position
            element_z = np.zeros((self.n_rx, 1))
        else:
            element_angle = element_position / probe_model.curvature_radius
            element_x = probe_model.curvature_radius * np.sin(element_angle)
            element_z = probe_model.curvature_radius * (
                    np.cos(element_angle) - 1)

        tx_distance = radial_distance
        rx_distance = np.sqrt(
            (x_distance - element_x) ** 2 + (z_distance - element_z) ** 2)

        self.t = (tx_distance + rx_distance) / c + initial_delay
        self.delays = self.t * fs  # in number of samples
        total_n_samples = self.n_rx * self.n_samples
        # Move samples outside the available area
        self.delays[np.isclose(self.delays, self.n_samples-1)] = self.n_samples-1
        self.delays[self.delays > self.n_samples-1] = total_n_samples + 1
        # (RF data will also be unrolled to a vect. n_rx*n_samples elements,
        #  row-wise major order).
        self.delays = self.xp.asarray(self.delays)
        self.delays += self.xp.arange(0, self.n_rx).reshape(self.n_rx, 1) \
                       * self.n_samples
        self.delays = self.delays.reshape(-1, self.n_samples * self.n_rx) \
            .astype(self.xp.float32)
        # Apodization
        lambd = c / fc
        max_tang = math.tan(
            math.asin(min(1, 2 / 3 * lambd / probe_model.pitch)))
        rx_tang = np.abs(np.tan(np.arctan2(x_distance - element_x,
                                           z_distance - element_z) - element_angle))
        rx_apodization = (rx_tang < max_tang).astype(np.float32)
        rx_apod_sum = np.sum(rx_apodization, axis=0)
        rx_apod_sum[rx_apod_sum == 0] = 1
        rx_apodization = rx_apodization/(rx_apod_sum.reshape(1, self.n_samples))
        self.rx_apodization = self.xp.asarray(rx_apodization)
        # IQ correction
        self.t = self.xp.asarray(self.t)
        self.iq_correction = self.xp.exp(1j * 2 * np.pi * fc * self.t) \
            .astype(self.xp.complex64)
        # Create new output shape
        return const_metadata.copy(input_shape=(self.n_seq, self.n_tx, self.n_samples))

    def process(self, data):
        data = data.copy().reshape(self.n_seq*self.n_tx, self.n_rx * self.n_samples)

        self.interp1d_func(data, self.delays, self.buffer)
        out = self.buffer.reshape((self.n_seq, self.n_tx, self.n_rx, self.n_samples))
        if self.is_iq:
            out = out * self.iq_correction
        out = out * self.rx_apodization
        out = self.xp.sum(out, axis=2)
        return out.reshape((self.n_seq, self.n_tx, self.n_samples))


class EnvelopeDetection(Operation):
    """
    Envelope detection (Hilbert transform).

    Currently this op works only for I/Q data (complex64).
    """

    def __init__(self, num_pkg=None):
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):

        n_samples = const_metadata.input_shape[-1]
        if n_samples == 0:
            raise ValueError("Empty array is not accepted.")
        return const_metadata.copy(is_iq_data=False, dtype="float32")

    def process(self, data):
        if data.dtype != self.xp.complex64:
            raise ValueError(
                f"Data type {data.dtype} is currently not supported.")
        return self.xp.abs(data)


class Transpose(Operation):
    """
    Data transposition.
    """

    def __init__(self, axes=None):
        """
        :param axes: permutation of axes to apply
        """
        self.axes = axes
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        axes = list(range(len(input_shape)))[::-1] if self.axes is None else self.axes
        output_shape = tuple(input_shape[ax] for ax in axes)
        return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        return self.xp.transpose(data, self.axes)


class ScanConversion(Operation):
    """
    Scan conversion (interpolation to target mesh).

    Currently linear interpolation is used by default, values outside
    the input mesh will be set to 0.0.

    Currently the op is implement for CPU only.

    Currently, the op is available only for convex probes.
    """

    def __init__(self, x_grid, z_grid):
        """
        Scan converter constructor.

        :param x_grid: a vector of grid points along OX axis [m]
        :param z_grid: a vector of grid points along OZ axis [m]
        """
        self.dst_points = None
        self.dst_shape = None
        self.x_grid = x_grid.reshape(1, -1)
        self.z_grid = z_grid.reshape(1, -1)
        self.is_gpu = False
        self.num_pkg = None

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg != np:
            self.is_gpu = True
        self.num_pkg = num_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        probe = const_metadata.context.device.probe.model
        if probe.is_convex_array():
            self.process = self._process_convex
            return self._prepare_convex(const_metadata)
        else:
            # linear array or phased array
            seq = const_metadata.context.sequence
            # Determine scanning type based on the sequence of parameters.
            tx_centers = seq.tx_aperture_center_element
            if tx_centers is None:
                tx_centers = seq.tx_aperture_center
            tx_centers = set(np.atleast_1d(tx_centers))
            tx_angles = set(np.atleast_1d(seq.angles))
            # Phased array scanning:
            # - single TX/RX aperture position
            # - multiple different angles
            if len(tx_centers) == 1 and len(tx_angles) > 1:
                self.process = self._process_phased_array
                return self._prepare_phased_array(const_metadata)
            # Linear array scanning:
            # - single transmit angle (equal 0)
            # - multiple different aperture positions
            elif len(tx_centers) > 1 and len(tx_angles) == 1:
                self.process = self._process_linear_array
                return self._prepare_linear_array(const_metadata)
            else:
                raise ValueError("The given combination of TX/RX parameters is "
                                 "not supported by ScanConversion")

    def _prepare_linear_array(self, const_metadata: arrus.metadata.ConstMetadata):
        # Determine interpolation function.
        if self.num_pkg == np:
            raise ValueError("Currently scan conversion for linear array "
                             "probe is implemented only for GPU devices.")
        import cupy as cp
        import cupyx.scipy.ndimage
        self.interp_function = cupyx.scipy.ndimage.map_coordinates
        self.n_frames, n_samples, n_scanlines = const_metadata.input_shape
        seq = const_metadata.context.sequence
        if not isinstance(seq, arrus.ops.imaging.LinSequence):
            raise ValueError("Scan conversion works only with LinSequence.")
        medium = const_metadata.context.medium
        probe = const_metadata.context.device.probe.model
        tx_rx_params = arrus.kernels.imaging.preprocess_sequence_parameters(probe, seq)
        tx_aperture_center_element = tx_rx_params["tx_ap_cent"]
        n_elements = probe.n_elements
        if n_elements % 2 != 0:
            raise ValueError("Even number of probe elements is required.")
        pitch = probe.pitch
        data_desc = const_metadata.data_description
        c = _get_speed_of_sound(const_metadata.context)
        tx_center_diff = np.diff(tx_aperture_center_element)
        # Check if tx aperture centers are evenly spaced.
        if not np.allclose(tx_center_diff, [tx_center_diff[0]]*len(tx_center_diff)):
            raise ValueError("Transmits should be done by consecutive "
                             "center elements (got tx center elements: "
                             f"{tx_aperture_center_element}")
        tx_center_diff = tx_center_diff[0]
        # Determine input grid.
        input_x_grid_diff = tx_center_diff*pitch
        input_x_grid_origin = (tx_aperture_center_element[0]-(n_elements-1)/2)*pitch
        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = data_desc.sampling_frequency
        start_sample = seq.rx_sample_range[0]
        input_z_grid_origin = start_sample/acq_fs*c/2
        input_z_grid_diff = c/(fs*2)
        # Map x_grid and z_grid to the RF frame coordinates.
        interp_x_grid = (self.x_grid-input_x_grid_origin)/input_x_grid_diff
        interp_z_grid = (self.z_grid-input_z_grid_origin)/input_z_grid_diff
        self._interp_mesh = cp.asarray(np.meshgrid(interp_z_grid, interp_x_grid, indexing="ij"))

        self.dst_shape = self.n_frames, len(self.z_grid.squeeze()), len(self.x_grid.squeeze())
        self.buffer = cp.zeros(self.dst_shape, dtype=cp.float32)
        return const_metadata.copy(input_shape=self.dst_shape)

    def _process_linear_array(self, data):
        for i in range(self.n_frames):
            self.buffer[i] = self.interp_function(data[i], self._interp_mesh, order=1)
        return self.buffer

    def _prepare_convex(self, const_metadata: arrus.metadata.ConstMetadata):
        if self.num_pkg is np:
            self.interpolator = scipy.ndimage.map_coordinates
        else:
            import cupyx.scipy.ndimage
            self.interpolator = cupyx.scipy.ndimage.map_coordinates
        probe = const_metadata.context.device.probe.model
        medium = const_metadata.context.medium
        data_desc = const_metadata.data_description

        if not self.num_pkg == np:
            import cupy as cp
            self.x_grid = self.num_pkg.asarray(self.x_grid).astype(cp.float32)
            self.z_grid = self.num_pkg.asarray(self.z_grid).astype(cp.float32)

        self.n_frames, n_samples, n_scanlines = const_metadata.input_shape
        seq = const_metadata.context.sequence

        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = data_desc.sampling_frequency

        start_sample = seq.rx_sample_range[0]

        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound

        tx_ap_cent_ang, _, _ = arrus.kernels.imaging.get_aperture_center(
            seq.tx_aperture_center_element, probe)

        z_grid_moved = self.z_grid.T + probe.curvature_radius \
            - self.num_pkg.max(probe.element_pos_z)

        self.radGridIn = (
                (start_sample / acq_fs + self.num_pkg.arange(0, n_samples) / fs)
                * c / 2)

        self.azimuthGridIn = tx_ap_cent_ang
        azimuthGridOut = self.num_pkg.arctan2(self.x_grid, z_grid_moved)
        radGridOut = (self.num_pkg.sqrt(self.x_grid ** 2 + z_grid_moved ** 2)
                      - probe.curvature_radius)

        self.dst_shape = self.n_frames, len(self.z_grid.squeeze()), len(self.x_grid.squeeze())

        dst_points = self.num_pkg.dstack((radGridOut, azimuthGridOut))
        dst_points = self.num_pkg.transpose(dst_points, axes=(2, 0, 1))

        def get_equalized_diff(values, param_name):
            diffs = np.diff(values)
            # Check if all values are evenly spaced
            if not np.allclose(diffs, [diffs[0]]*len(diffs)):
                raise ValueError(f"{param_name} should be evenly spaced, "
                                 f"got {values}")
            return diffs[0]

        dst_points[0] -= self.radGridIn[0]
        dst_points[0] /= get_equalized_diff(self.radGridIn,
                                            "Input radial distance")
        dst_points[1] -= self.azimuthGridIn[0]
        dst_points[1] /= get_equalized_diff(self.azimuthGridIn,
                                            "Azimuth angle")
        self.dst_points = self.num_pkg.asarray(dst_points,
                                               dtype=self.num_pkg.float32)
        self.output_buffer = self.num_pkg.zeros(self.dst_shape, dtype=np.float32)
        return const_metadata.copy(input_shape=self.dst_shape)

    def _process_convex(self, data):
        data[np.isnan(data)] = 0.0
        # TODO do batch-wise processing here
        for i in range(self.n_frames):
            self.output_buffer[i] = self.interpolator(data[i],
                                                      self.dst_points,
                                                      order=1)
        return self.output_buffer

    def _prepare_phased_array(self, const_metadata: arrus.metadata.ConstMetadata):
        probe = const_metadata.context.device.probe.model
        data_desc = const_metadata.data_description

        self.n_frames, n_samples, n_scanlines = const_metadata.input_shape
        seq = const_metadata.context.sequence
        fs = const_metadata.context.device.sampling_frequency
        acq_fs = fs / seq.downsampling_factor
        fs = data_desc.sampling_frequency
        start_sample, _ = seq.rx_sample_range
        start_time = start_sample/acq_fs
        c = _get_speed_of_sound(const_metadata.context)
        tx_rx_params = arrus.kernels.imaging.preprocess_sequence_parameters(probe, seq)
        tx_ap_cent_elem = np.array(tx_rx_params["tx_ap_cent"])[0]
        tx_ap_cent_ang, tx_ap_cent_x, tx_ap_cent_z = arrus.kernels.imaging.get_aperture_center(
            tx_ap_cent_elem, probe)

        # There is a single position of TX aperture.
        tx_ap_cent_x = tx_ap_cent_x.squeeze().item()
        tx_ap_cent_z = tx_ap_cent_z.squeeze().item()
        tx_ap_cent_ang = tx_ap_cent_ang.squeeze().item()

        self.radGridIn = (start_time + np.arange(0, n_samples)/fs)*c/2
        self.azimuthGridIn = seq.angles + tx_ap_cent_ang
        azimuthGridOut = np.arctan2((self.x_grid-tx_ap_cent_x), (self.z_grid.T-tx_ap_cent_z))
        radGridOut = np.sqrt((self.x_grid-tx_ap_cent_x)**2 + (self.z_grid.T-tx_ap_cent_z)**2)
        dst_points = np.dstack((radGridOut, azimuthGridOut))
        w, h, d = dst_points.shape
        self.dst_points = dst_points.reshape((w * h, d))
        self.dst_shape = self.n_frames, len(self.z_grid.squeeze()), len(self.x_grid.squeeze())
        self.output_buffer = np.zeros(self.dst_shape, dtype=np.float32)
        return const_metadata.copy(input_shape=self.dst_shape)

    def _process_phased_array(self, data):
        if self.is_gpu:
            data = data.get()
        data[np.isnan(data)] = 0.0
        for i in range(self.n_frames):
            self.interpolator = scipy.interpolate.RegularGridInterpolator(
                (self.radGridIn, self.azimuthGridIn), data[i], method="linear",
                bounds_error=False, fill_value=0)
            result = self.interpolator(self.dst_points).reshape(self.dst_shape[1:])
            self.output_buffer[i] = result
        return self.num_pkg.asarray(self.output_buffer).astype(np.float32)


class LogCompression(Operation):
    """
    Converts data to decibel scale.
    """
    def __init__(self):
        self.num_pkg = None
        self.is_gpu = False

    def set_pkgs(self, num_pkg, **kwargs):
        self.num_pkg = num_pkg
        if self.num_pkg != np:
            self.is_gpu = True

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        n_samples = const_metadata.input_shape[-1]
        if n_samples == 0:
            raise ValueError("Empty array is not accepted.")
        return const_metadata

    def process(self, data):
        data[data <= 0] = 1e-9
        return 20*self.num_pkg.log10(data)


class DynamicRangeAdjustment(Operation):
    """
    Clips data values to given range.
    """

    def __init__(self, min=20, max=80):
        """
        Constructor.

        :param min: minimum value to clamp
        :param max: maximum value to clamp
        """
        self.min = min
        self.max = max
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def process(self, data):
        return self.xp.clip(data, a_min=self.min, a_max=self.max)


class ToGrayscaleImg(Operation):
    """
    Converts data to grayscale image (uint8).
    """

    def __init__(self):
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def process(self, data):
        data = data - self.xp.min(data)
        data = data/self.xp.max(data)*255
        return data.astype(self.xp.uint8)


class Enqueue(Operation):
    """
    Copies the output data from the previous stage to the provided queue
    and passes the input data unmodified for further processing.

    Works on queue.Queue objects.

    :param queue: queue.Queue instance
    :param block: if true, will block the pipeline until a free slot is available,
      otherwise will raise queue.Full exception
    :param ignore_full: when true and block = False, this step will not throw
      queue.Full exception. Set it to true if it is ok to ommit some of the
      acquired frames.
    """

    def __init__(self, queue, block=True, ignore_full=False):
        self.queue = queue
        self.block = block
        self.ignore_full = ignore_full
        if self.block:
            self.process = self._put_block
        else:
            if self.ignore_full:
                self.process = self._put_ignore_full
            else:
                self.process = self._put_non_block
        self._copy_func = None

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg == np:
            self._copy_func = np.copy
        else:
            import cupy as cp
            self._copy_func = cp.asnumpy

    def prepare(self, const_metadata):
        return const_metadata

    def initialize(self, data):
        # data.get()
        return data

    def _put_block(self, data):
        self.queue.put(self._copy_func(data))
        return data

    def _put_non_block(self, data):
        self.queue.put_nowait(self._copy_func(data))
        return data

    def _put_ignore_full(self, data):
        try:
            self.queue.put_nowait(self._copy_func(data))
        except queue.Full:
            pass
        return data


class SelectSequenceRaw(Operation):

    def __init__(self, sequence):
        if isinstance(sequence, Iterable) and len(sequence) > 1:
            raise ValueError("Only a single sequence can be selected")
        self.sequence = sequence
        self.output = None
        self.num_pkg = None
        self.positions = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.num_pkg = num_pkg

    def prepare(self, const_metadata):
        context = const_metadata.context
        seq = context.sequence
        raw_seq = context.raw_sequence
        n_seq = len(self.sequence)

        # For each us4oem, compute tuples: (src_start, dst_start, src_end, dst_end)
        # Where each value is the number of rows (we assume 32 columns, i.e. RX channels)
        n_samples_set = {op.rx.get_n_samples() for op in raw_seq.ops}

        if len(n_samples_set) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                f"Each tx/rx in the sequence should acquire the same number of "
                f"samples (actual: {n_samples_set})")
        n_samples = next(iter(n_samples_set))

        fcm = const_metadata.data_description.custom["frame_channel_mapping"]
        fcm_us4oems = fcm.us4oems
        fcm_frames = fcm.frames
        # TODO update frame offsets
        us4oems = set(fcm.us4oems.flatten().tolist())
        sorted(us4oems)

        self.positions = []
        dst_start = 0
        dst_end = 0
        frame_offsets = []
        current_frame = 0  # Current physical frame.
        for us4oem in us4oems:
            n_frames = self.num_pkg.max(fcm_frames[fcm_us4oems == us4oem])+1
            us4oem_offset = fcm.frame_offsets[us4oem]
            # NOTE: below we use only a single sequence
            src_start = us4oem_offset*n_samples+self.sequence[0]*n_frames*n_samples
            src_end = src_start+n_frames*n_samples
            dst_end = dst_start+n_frames*n_samples
            self.positions.append((src_start, dst_start, src_end, dst_end))
            frame_offsets.append(current_frame)
            current_frame += n_frames
            dst_start = dst_end

        output_shape = (dst_end, 32)
        self.output = self.num_pkg.zeros(output_shape, dtype=np.int16)

        # Update const metadata
        new_seq = dataclasses.replace(seq, n_repeats=n_seq)
        new_raw_seq = dataclasses.replace(raw_seq, n_repeats=n_seq)
        new_context = arrus.metadata.FrameAcquisitionContext(
            device=context.device, sequence=new_seq,
            raw_sequence=new_raw_seq, medium=context.medium,
            custom_data=context.custom_data)

        # Update FCM (change the batch_size)
        data_desc = const_metadata.data_description
        data_desc_custom = data_desc.custom
        new_data_desc_custom = data_desc_custom.copy()
        fcm = data_desc_custom["frame_channel_mapping"]
        new_fcm = dataclasses.replace(fcm, batch_size=1,
                                      frame_offsets=frame_offsets)
        new_data_desc_custom["frame_channel_mapping"] = new_fcm
        new_data_desc = dataclasses.replace(data_desc, custom=new_data_desc_custom)

        return const_metadata.copy(input_shape=output_shape,
                                   context=new_context,
                                   data_desc=new_data_desc)

    def process(self, data):
        for src_start, dst_start, src_end, dst_end in self.positions:
            self.output[dst_start:dst_end, :] = data[src_start:src_end, :]
        return self.output


class SelectSequence(Operation):
    """
    Selects sequences for a given batch for further processing.

    This operator modifies input context so the appropriate
    number of sequences is properly set.

    :param frames: sequences to select
    """

    def __init__(self, sequence):
        if not isinstance(sequence, Iterable):
            # Wrap into an array
            sequence = [sequence]
        self.sequence = sequence

    def set_pkgs(self, **kwargs):
        pass

    def prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        context = const_metadata.context
        seq = context.sequence
        raw_seq = context.raw_sequence
        n_seq = len(self.sequence)

        output_shape = input_shape[1:]
        output_shape = (n_seq, ) + output_shape
        new_seq = dataclasses.replace(seq, n_repeats=n_seq)
        new_raw_seq = dataclasses.replace(raw_seq, n_repeats=n_seq)
        new_context = arrus.metadata.FrameAcquisitionContext(
            device=context.device, sequence=new_seq,
            raw_sequence=new_raw_seq, medium=context.medium,
            custom_data=context.custom_data)
        return const_metadata.copy(input_shape=output_shape,
                                   context=new_context)

    def process(self, data):
        return data[self.sequence]


class SelectFrames(Operation):
    """
    Selects frames for a given sequence for further processing.
    """

    def __init__(self, frames):
        """
        Constructor.

        :param frames: frames to select
        """
        self.frames = frames

    def set_pkgs(self, **kwargs):
        pass

    def prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        context = const_metadata.context
        seq = context.sequence
        n_frames = len(self.frames)

        if len(input_shape) != 3:
            raise ValueError("The input should be 3-D "
                             "(frame number should be the first axis)")

        input_n_frames, d2, d3 = input_shape
        output_shape = (n_frames, d2, d3)
        # TODO make this op less prone to changes in op implementation
        if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
            # select appropriate angles
            angles = self._limit_params(seq.angles, self.frames)
            tx_focus = self._limit_params(seq.tx_focus, self.frames)
            tx_aperture_center_element = self._limit_params(seq.tx_aperture_center_element,
                                                            self.frames)
            tx_aperture_center = self._limit_params(seq.tx_aperture_center, self.frames)
            rx_aperture_center_element = self._limit_params(seq.rx_aperture_center_element,
                                                            self.frames)
            rx_aperture_center = self._limit_params(seq.rx_aperture_center, self.frames)

            new_seq = dataclasses.replace(
                seq,
                angles=angles,
                tx_focus=tx_focus,
                tx_aperture_center_element=tx_aperture_center_element,
                tx_aperture_center=tx_aperture_center,
                rx_aperture_center_element=rx_aperture_center_element,
                rx_aperture_center=rx_aperture_center)
            new_context = const_metadata.context
            new_context = arrus.metadata.FrameAcquisitionContext(
                device=new_context.device, sequence=new_seq,
                raw_sequence=new_context.raw_sequence,
                medium=new_context.medium, custom_data=new_context.custom_data)
            return const_metadata.copy(input_shape=output_shape,
                                       context=new_context)
        else:
            return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        return data[self.frames]

    def _limit_params(self, value, frames):
        if value is not None and hasattr(value, "__len__") and len(value) > 1:
            return np.array(value)[frames]
        else:
            return value


# Alias
SelectFrame = SelectFrames


class Squeeze(Operation):
    """
    Squeezes input array (removes axes = 1).
    """
    def __init__(self):
        pass

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata):
        output_shape = tuple(i for i in const_metadata.input_shape if i != 1)
        return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        return self.xp.squeeze(data)


class ReconstructLri(Operation):
    """
    Rx beamforming for synthetic aperture imaging.

    Expected input data shape: n_emissions, n_rx, n_samples
    :param x_grid: output image grid points (OX coordinates)
    :param z_grid: output image grid points  (OZ coordinates)
    :param rx_tang_limits: RX apodization angle limits (given as the tangent of the angle), \
      a pair of values (min, max). If not provided or None, [-0.5, 0.5] range will be used
    """

    def __init__(self, x_grid, z_grid, rx_tang_limits=None):
        self.x_grid = x_grid
        self.z_grid = z_grid
        import cupy as cp
        self.num_pkg = cp
        self.rx_tang_limits = rx_tang_limits # Currently used only by Convex PWI implementation

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg is np:
            raise ValueError("ReconstructLri operation is implemented for GPU only.")

    def prepare(self, const_metadata):
        import cupy as cp

        current_dir = os.path.dirname(os.path.join(os.path.abspath(__file__)))
        _kernel_source = Path(os.path.join(current_dir, "iq_raw_2_lri.cu")).read_text()
        self._kernel_module = self.num_pkg.RawModule(code=_kernel_source)
        self._kernel = self._kernel_module.get_function("iqRaw2Lri")
        self._z_elem_const = self._kernel_module.get_global("zElemConst")
        self._tang_elem_const = self._kernel_module.get_global("tangElemConst")

        # INPUT PARAMETERS.
        # Input data shape.
        self.n_seq, self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape

        seq = const_metadata.context.sequence
        probe_model = const_metadata.context.device.probe.model
        acq_fs = (const_metadata.context.device.sampling_frequency / seq.downsampling_factor)
        start_sample = seq.rx_sample_range[0]

        self.x_size = len(self.x_grid)
        self.z_size = len(self.z_grid)
        output_shape = (self.n_seq, self.n_tx, self.x_size, self.z_size)
        self.output_buffer = self.num_pkg.zeros(output_shape, dtype=self.num_pkg.complex64)
        x_block_size = min(self.x_size, 16)
        z_block_size = min(self.z_size, 16)
        tx_block_size = min(self.n_tx, 4)
        self.block_size = (z_block_size, x_block_size, tx_block_size)
        self.grid_size = (int((self.z_size-1)//z_block_size + 1),
                          int((self.x_size-1)//x_block_size + 1),
                          int((self.n_seq*self.n_tx-1)//tx_block_size + 1))
        self.x_pix = self.num_pkg.asarray(self.x_grid, dtype=self.num_pkg.float32)
        self.z_pix = self.num_pkg.asarray(self.z_grid, dtype=self.num_pkg.float32)

        # System and transmit properties.
        self.sos = self.num_pkg.float32(seq.speed_of_sound)
        self.fs = self.num_pkg.float32(const_metadata.data_description.sampling_frequency)
        self.fn = self.num_pkg.float32(seq.pulse.center_frequency)
        self.pitch = self.num_pkg.float32(probe_model.pitch)

        # Probe description
        element_pos_x = probe_model.element_pos_x
        element_pos_z = probe_model.element_pos_z
        element_angle_tang = np.tan(probe_model.element_angle)

        self.n_elements = probe_model.n_elements

        device_props = cp.cuda.runtime.getDeviceProperties(0)
        if device_props["totalConstMem"] < 256*3*4:  # 3 float32 arrays, 256 elements max
            raise ValueError("There is not enough constant memory available!")

        x_elem = np.asarray(element_pos_x, dtype=self.num_pkg.float32)
        self._x_elem_const = _get_const_memory_array(
            self._kernel_module, name="xElemConst", input_array=x_elem)
        z_elem = np.asarray(element_pos_z, dtype=self.num_pkg.float32)
        self._z_elem_const = _get_const_memory_array(
            self._kernel_module, name="zElemConst", input_array=z_elem)
        tang_elem = np.asarray(element_angle_tang, dtype=self.num_pkg.float32)
        self._tang_elem_const = _get_const_memory_array(
            self._kernel_module, name="tangElemConst", input_array=tang_elem)

        # TX aperture description
        # Convert the sequence to the positions of the aperture centers
        tx_rx_params = arrus.kernels.imaging.compute_tx_rx_params(
            probe_model,
            seq,
            seq.speed_of_sound)
        tx_centers, tx_sizes = tx_rx_params["tx_ap_cent"], tx_rx_params["tx_ap_size"]
        rx_centers, rx_sizes = tx_rx_params["rx_ap_cent"], tx_rx_params["rx_ap_size"]
        tx_center_delay = tx_rx_params["tx_center_delay"]

        tx_center_angles, tx_center_x, tx_center_z = arrus.kernels.imaging.get_aperture_center(tx_centers, probe_model)
        tx_center_angles = tx_center_angles + seq.angles
        self.tx_ang_zx = self.num_pkg.asarray(tx_center_angles, dtype=self.num_pkg.float32)
        self.tx_ap_cent_x = self.num_pkg.asarray(tx_center_x, dtype=self.num_pkg.float32)
        self.tx_ap_cent_z = self.num_pkg.asarray(tx_center_z, dtype=self.num_pkg.float32)

        # first/last probe element in TX aperture
        tx_ap_origin = np.round(tx_centers-(tx_sizes-1)/2 + 1e-9).astype(np.int32)
        rx_ap_origin = np.round(rx_centers-(rx_sizes-1)/2 + 1e-9).astype(np.int32)
        tx_ap_first_elem = np.maximum(tx_ap_origin, 0)
        tx_ap_last_elem = np.minimum(tx_ap_origin+tx_sizes-1, probe_model.n_elements-1)
        self.tx_ap_first_elem = self.num_pkg.asarray(tx_ap_first_elem, dtype=self.num_pkg.int32)
        self.tx_ap_last_elem = self.num_pkg.asarray(tx_ap_last_elem, dtype=self.num_pkg.int32)
        self.rx_ap_origin = self.num_pkg.asarray(rx_ap_origin, dtype=self.num_pkg.int32)

        # Min/max tang
        if self.rx_tang_limits is not None:
            self.min_tang, self.max_tang = self.rx_tang_limits
        else:
            # Default:
            self.min_tang, self.max_tang = -0.5, 0.5

        self.min_tang = self.num_pkg.float32(self.min_tang)
        self.max_tang = self.num_pkg.float32(self.max_tang)
        self.tx_foc = self.num_pkg.asarray([seq.tx_focus]*self.n_tx, dtype=self.num_pkg.float32)
        burst_factor = seq.pulse.n_periods / (2*self.fn)
        self.initial_delay = -start_sample/65e6+burst_factor+tx_center_delay
        self.initial_delay = self.num_pkg.float32(self.initial_delay)
        return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        data = self.num_pkg.ascontiguousarray(data)
        params = (
            self.output_buffer,
            data,
            self.n_elements,
            self.n_seq, self.n_tx, self.n_samples,
            self.z_pix, self.z_size,
            self.x_pix, self.x_size,
            self.sos, self.fs, self.fn,
            self.tx_foc, self.tx_ang_zx,
            self.tx_ap_cent_z, self.tx_ap_cent_x,
            self.tx_ap_first_elem, self.tx_ap_last_elem,
            self.rx_ap_origin, self.n_rx,
            self.min_tang, self.max_tang,
            self.initial_delay)
        self._kernel(self.grid_size, self.block_size, params)
        return self.output_buffer


class Sum(Operation):
    """
    Sum of array elements over a given axis.

    :param axis: axis along which a sum is performed
    """

    def __init__(self, axis=-1):
        self.axis = axis
        self.num_pkg = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.num_pkg = num_pkg

    def prepare(self, const_metadata):
        output_shape = list(const_metadata.input_shape)
        actual_axis = len(output_shape)-1 if self.axis == -1 else self.axis
        del output_shape[actual_axis]
        return const_metadata.copy(input_shape=tuple(output_shape))

    def process(self, data):
        return self.num_pkg.sum(data, axis=self.axis)


class Mean(Operation):
    """
    Average of array elements over a given axis.

    :param axis: axis along which a average is computed
    """

    def __init__(self, axis=-1):
        self.axis = axis
        self.num_pkg = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.num_pkg = num_pkg

    def prepare(self, const_metadata):
        output_shape = list(const_metadata.input_shape)
        actual_axis = len(output_shape)-1 if self.axis == -1 else self.axis
        del output_shape[actual_axis]
        return const_metadata.copy(input_shape=tuple(output_shape))

    def process(self, data):
        return self.num_pkg.mean(data, axis=self.axis)


def _get_rx_aperture_origin(aperture_center_element, aperture_size):
    return np.round(aperture_center_element-(aperture_size-1)/2+1e-9)


# -------------------------------------------- RF frame remapping.
# CPU remapping tools.
@dataclasses.dataclass
class Transfer:
    src_frame: int
    src_range: tuple
    dst_frame: int
    dst_range: tuple


def __group_transfers(frame_channel_mapping):
    result = []
    frame_mapping = frame_channel_mapping.frames
    channel_mapping = frame_channel_mapping.channels
    if frame_mapping.size == 0 or channel_mapping.size == 0:
        raise RuntimeError("Empty frame channel mappings")
    # Number of logical frames
    n_frames, n_channels = channel_mapping.shape
    for dst_frame in range(n_frames):
        current_dst_range = None
        prev_src_frame = None
        prev_src_channel = None
        current_src_frame = None
        current_src_range = None
        for dst_channel in range(n_channels):
            src_frame = frame_mapping[dst_frame, dst_channel]
            src_channel = channel_mapping[dst_frame, dst_channel]
            if src_channel < 0:
                # Omit current channel.
                # Negative src channel means, that the given channel
                # is not available and should be treated as missing.
                continue
            if (prev_src_frame is None  # the first transfer
                    # new src frame
                    or src_frame != prev_src_frame
                    # a gap in current frame
                    or src_channel != prev_src_channel+1):
                # Close current source range
                if current_src_frame is not None:
                    transfer = Transfer(
                        src_frame=current_src_frame,
                        src_range=tuple(current_src_range),
                        dst_frame=dst_frame,
                        dst_range=tuple(current_dst_range)
                    )
                    result.append(transfer)
                # Start a new range
                current_src_frame = src_frame
                # [start, end)
                current_src_range = [src_channel, src_channel + 1]
                current_dst_range = [dst_channel, dst_channel + 1]
            else:
                # Continue current range
                current_src_range[1] = src_channel + 1
                current_dst_range[1] = dst_channel + 1
            prev_src_frame = src_frame
            prev_src_channel = src_channel
        # End a range for current frame.
        current_src_range = int(current_src_range[0]), int(current_src_range[1])
        transfer = Transfer(
            src_frame=int(current_src_frame),
            src_range=tuple(current_src_range),
            dst_frame=dst_frame,
            dst_range=tuple(current_dst_range))
        result.append(transfer)
    return result


def __remap(output_array, input_array, transfers):
    input_array = input_array
    for t in transfers:
        dst_l, dst_r = t.dst_range
        src_l, src_r = t.src_range
        output_array[t.dst_frame, :, dst_l:dst_r] = \
            input_array[t.src_frame, :, src_l:src_r]


class RemapToLogicalOrder(Operation):
    """
    Remaps the order of the data to logical order defined by the us4r device.

    If the batch size was equal 1, the raw ultrasound RF data with shape.
    (n_frames, n_samples, n_channels).
    A single metadata object will be returned.

    If the batch size was > 1, the the raw ultrasound RF data with shape
    (n_us4oems*n_samples*n_frames*n_batches, 32) will be reordered to
    (batch_size, n_frames, n_samples, n_channels). A list of metadata objects
    will be returned.
    """

    def __init__(self, num_pkg=None):
        self._transfers = None
        self._output_buffer = None
        self.xp = num_pkg
        self.remap = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _is_prepared(self):
        return self._transfers is not None and self._output_buffer is not None

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        xp = self.xp
        # get shape, create an array with given shape
        # create required transfers
        # perform the transfers
        fcm = const_metadata.data_description.custom["frame_channel_mapping"]
        n_frames, n_channels = fcm.frames.shape
        n_samples_set = {op.rx.get_n_samples()
                         for op in const_metadata.context.raw_sequence.ops}

        # get (unique) number of samples in a frame
        if len(n_samples_set) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                f"Each tx/rx in the sequence should acquire the same number of "
                f"samples (actual: {n_samples_set})")
        n_samples = next(iter(n_samples_set))
        batch_size = fcm.batch_size
        self.output_shape = (batch_size, n_frames, n_samples, n_channels)
        self._output_buffer = xp.zeros(shape=self.output_shape, dtype=xp.int16)
        if xp == np:
            # CPU
            raise ValueError(f"'{type(self).__name__}' is not implemented for CPU")
        else:
            # GPU
            import cupy as cp
            from arrus.utils.us4r_remap_gpu import get_default_grid_block_size, run_remap
            self._fcm_frames = cp.asarray(fcm.frames)
            self._fcm_channels = cp.asarray(fcm.channels)
            self._fcm_us4oems = cp.asarray(fcm.us4oems)
            frame_offsets = fcm.frame_offsets
            #  TODO constant memory
            self._frame_offsets = cp.asarray(frame_offsets)
            # For each us4OEM, get number of physical frames this us4OEM gathers.
            # Note: this is the max number of us4OEMs IN USE.
            n_us4oems = cp.max(self._fcm_us4oems).get()+1
            n_frames_us4oems = []
            for us4oem in range(n_us4oems):
                us4oem_frames = self._fcm_frames[self._fcm_us4oems == us4oem]
                if us4oem_frames.size == 0:
                    n_frames_us4oems.append(0)
                else:
                    n_frames_us4oem = cp.max(us4oem_frames).get().item()
                    n_frames_us4oems.append(n_frames_us4oem)

            #  TODO constant memory
            self._n_frames_us4oems = cp.asarray(n_frames_us4oems, dtype=cp.uint32)+1
            self.grid_size, self.block_size = get_default_grid_block_size(
                self._fcm_frames, n_samples,
                batch_size
            )
            def gpu_remap_fn(data):
                run_remap(self.grid_size, self.block_size,
                    [self._output_buffer, data,
                     self._fcm_frames, self._fcm_channels, self._fcm_us4oems,
                     self._frame_offsets,
                     self._n_frames_us4oems,
                     batch_size, n_frames, n_samples, n_channels])

            self._remap_fn = gpu_remap_fn
        return const_metadata.copy(input_shape=self.output_shape)

    def process(self, data):
        self._remap_fn(data)
        return self._output_buffer


class Reshape(Operation):
    """
    Reshapes input data to a given shape.
    """

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def prepare(self, const_metadata):
        return const_metadata.copy(input_shape=self.shape)

    def process(self, data):
        return data.reshape(*self.shape)


def _get_const_memory_array(module, name, input_array):
    import cupy as cp
    const_arr_ptr = module.get_global(name)
    const_arr = cp.ndarray(shape=input_array.shape, dtype=input_array.dtype,
                           memptr=const_arr_ptr)
    const_arr.set(input_array)
    return const_arr


def _read_kernel_module(path):
    import cupy as cp
    current_dir = os.path.dirname(os.path.join(os.path.abspath(__file__)))
    kernel_src = Path(os.path.join(current_dir, path)).read_text()
    return cp.RawModule(code=kernel_src)


def _get_speed_of_sound(context):
    seq = context.sequence
    medium = context.medium
    if seq.speed_of_sound is not None:
        return seq.speed_of_sound
    else:
        return medium.speed_of_sound


class ExtractMetadata(Operation):

    def __init__(self):
        super().__init__()

    def set_pkgs(self, **kwargs):
        super().set_pkgs(**kwargs)

    def prepare(self, const_metadata):
        n_samples = const_metadata.context.raw_sequence.get_n_samples()
        if len(n_samples) > 1:
            raise ValueError("All Rx ops should gather the same number "
                             "of samples.")
        self._n_samples = next(iter(n_samples))
        fcm = const_metadata.data_description.custom["frame_channel_mapping"]
        # Metadata is saved by us4OEM:0 module only.
        self._n_frames = fcm.n_frames[0]
        self._n_repeats = const_metadata.context.raw_sequence.n_repeats
        return const_metadata

    def process(self, data):
        return data[:self._n_samples*self._n_frames:self._n_samples]\
            .reshape(self._n_frames, -1) \
            .copy()


class ReconstructLri3D(Operation):
    """
    Rx beamforming for synthetic aperture imaging for matrix array.

    tx_foc, tx_ang_zx, tx_ang_zy: arrays

    Expected input data shape: batch_size, n_emissions, n_rx_x, n_rx_y, n_samples
    :param x_grid: output image grid points (OX coordinates)
    :param y_grid: output image grid points (OY coordinates)
    :param z_grid: output image grid points  (OZ coordinates)
    :param rx_tang_limits: RX apodization angle limits (given as the tangent of the angle), \
      a pair of values (min, max). If not provided or None, [-0.5, 0.5] range will be used
    """

    def __init__(self, x_grid, y_grid, z_grid, tx_foc, tx_ang_zx, tx_ang_zy,
                 speed_of_sound, rx_tang_limits=None):
        self.tx_ang_zy = tx_ang_zy
        self.tx_ang_zx = tx_ang_zx
        self.tx_foc = tx_foc
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        self.speed_of_sound = speed_of_sound
        import cupy as cp
        self.num_pkg = cp
        self.rx_tang_limits = rx_tang_limits

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg is np:
            raise ValueError("ReconstructLri3D operation is implemented for GPU only.")

    def _get_aperture_boundaries(self, apertures):
        def get_min_max_x_y(ap):
            cords = np.argwhere(ap)
            y, x = zip(*cords)
            return np.min(x), np.max(x), np.min(y), np.max(y)
        min_max_x_y = (get_min_max_x_y(aperture) for aperture in apertures)
        min_x, max_x, min_y, max_y = zip(*min_max_x_y)
        min_x, max_x = np.atleast_1d(min_x), np.atleast_1d(max_x)
        min_y, max_y = np.atleast_1d(min_y), np.atleast_1d(max_y)
        return min_x, max_x, min_y, max_y

    def prepare(self, const_metadata):
        import cupy as cp

        current_dir = os.path.dirname(os.path.join(os.path.abspath(__file__)))
        _kernel_source = Path(os.path.join(current_dir, "iq_raw_2_lri_3d.cu")).read_text()
        self._kernel_module = self.num_pkg.RawModule(code=_kernel_source)
        self._kernel_module.compile()
        self._kernel = self._kernel_module.get_function("iqRaw2Lri3D")

        # INPUT PARAMETERS.
        # Input data shape.
        self.n_seq, self.n_tx, self.n_rx_y, self.n_rx_x, self.n_samples = const_metadata.input_shape

        seq = const_metadata.context.raw_sequence
        # TODO note: we assume here that a single TX/RX has the below properties
        # the same for each TX/RX. Validation is missing here.
        ref_tx_rx = seq.ops[0]
        ref_tx = ref_tx_rx.tx
        ref_rx = ref_tx_rx.rx
        probe_model = const_metadata.context.device.probe.model
        acq_fs = (const_metadata.context.device.sampling_frequency / ref_rx.downsampling_factor)
        start_sample = ref_rx.sample_range[0]

        self.y_size = len(self.y_grid)
        self.x_size = len(self.x_grid)
        self.z_size = len(self.z_grid)
        output_shape = (self.n_seq, self.y_size, self.x_size, self.z_size)
        self.output_buffer = self.num_pkg.zeros(output_shape, dtype=self.num_pkg.complex64)
        x_block_size = min(self.z_size, 8)
        y_block_size = min(self.x_size, 8)
        z_block_size = min(self.y_size, 8)
        self.block_size = (x_block_size, y_block_size, z_block_size)
        self.grid_size = (int((self.z_size-1)//z_block_size + 1),
                          int((self.x_size-1)//x_block_size + 1),
                          int((self.y_size-1)//y_block_size + 1))

        self.y_pix = self.num_pkg.asarray(self.y_grid, dtype=self.num_pkg.float32)
        self.x_pix = self.num_pkg.asarray(self.x_grid, dtype=self.num_pkg.float32)
        self.z_pix = self.num_pkg.asarray(self.z_grid, dtype=self.num_pkg.float32)

        # System and transmit properties.
        self.sos = self.num_pkg.float32(self.speed_of_sound)
        self.fs = self.num_pkg.float32(const_metadata.data_description.sampling_frequency)
        self.fn = self.num_pkg.float32(ref_tx.excitation.center_frequency)
        self.tx_foc = self.num_pkg.asarray(self.tx_foc).astype(np.float32)
        self.tx_ang_zx = self.num_pkg.asarray(self.tx_ang_zx).astype(np.float32)
        self.tx_ang_zy = self.num_pkg.asarray(self.tx_ang_zy).astype(np.float32)

        # Probe description
        # TODO specific for Vermon mat-3d probe.
        probe_model = const_metadata.context.device.probe.model
        pitch = 0.3e-3 # probe_model.pitch
        self.n_elements = 32
        n_rows_x = self.n_elements
        n_rows_y = self.n_elements + 3
        # General regular position of elements
        element_pos_x = np.linspace(-(n_rows_x - 1)/2, (n_rows_x - 1)/2, num=n_rows_x)
        element_pos_x = element_pos_x*pitch

        element_pos_y = np.linspace(-(n_rows_y - 1)/2, (n_rows_y - 1)/2, num=n_rows_y)
        element_pos_y = element_pos_y*pitch
        element_pos_y = np.delete(element_pos_y, (8, 17, 26))

        element_pos_x = element_pos_x.astype(np.float32)
        element_pos_y = element_pos_y.astype(np.float32)
        # Put the data into GPU constant memory.
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        if device_props["totalConstMem"] < 256*2*4:  # 2 float32 arrays, 256 elements max
            raise ValueError("There is not enough constant memory available!")
        x_elem = np.asarray(element_pos_x, dtype=self.num_pkg.float32)
        self._x_elem_const = _get_const_memory_array(
            self._kernel_module, name="xElemConst", input_array=x_elem)
        y_elem = np.asarray(element_pos_y, dtype=self.num_pkg.float32)
        self._y_elem_const = _get_const_memory_array(
            self._kernel_module, name="yElemConst", input_array=y_elem)

        def get_min_max_x_y(aperture):
            cords = np.argwhere(aperture)
            y, x = zip(*cords)
            return np.min(x), np.max(x), np.min(y), np.max(y)

        # TODO assumption, that probe has the same number elements in both dimensions
        tx_apertures = (tx_rx.tx.aperture.reshape((self.n_elements, self.n_elements)) for tx_rx in seq.ops)
        rx_apertures = (tx_rx.rx.aperture.reshape((self.n_elements, self.n_elements)) for tx_rx in seq.ops)
        tx_bounds = self._get_aperture_boundaries(tx_apertures)
        rx_bounds = self._get_aperture_boundaries(rx_apertures)
        txap_min_x, txap_max_x, txap_min_y, txap_max_y = tx_bounds
        rxap_min_x, rxap_max_x, rxap_min_y, rxap_max_y = rx_bounds
        rxap_size_x = set((rxap_max_x-rxap_min_x).tolist())
        rxap_size_y = set((rxap_max_y-rxap_min_y).tolist())
        if len(rxap_size_x) > 1 or len(rxap_size_y) > 1:
            raise ValueError("Each TX/RX aperture should have the same square aperture size.")
        rxap_size_x = next(iter(rxap_size_x))
        rxap_size_y = next(iter(rxap_size_y))
        # The above can be also compared with the size of data, but right we are not doing it here

        self.tx_ap_first_elem_x = self.num_pkg.asarray(txap_min_x, dtype=self.num_pkg.int32)
        self.tx_ap_last_elem_x = self.num_pkg.asarray(txap_max_x, dtype=self.num_pkg.int32)
        self.tx_ap_first_elem_y = self.num_pkg.asarray(txap_min_y, dtype=self.num_pkg.int32)
        self.tx_ap_last_elem_y = self.num_pkg.asarray(txap_max_y, dtype=self.num_pkg.int32)
        # RX AP
        self.rx_ap_first_elem_x = self.num_pkg.asarray(rxap_min_x, dtype=self.num_pkg.int32)
        self.rx_ap_first_elem_y = self.num_pkg.asarray(rxap_min_y, dtype=self.num_pkg.int32)

        # Find the center of TX aperture.
        # TODO note: this method assumes that all TX/RXs have a rectangle TX aperture
        # 1. Find the position of the center.
        tx_ap_center_x = (element_pos_x[txap_min_x] + element_pos_x[txap_max_x])/2
        tx_ap_center_y = (element_pos_y[txap_min_y] + element_pos_y[txap_max_y])/2
        # element index -> element position
        ap_center_elem_x = np.interp(tx_ap_center_x, element_pos_x, np.arange(len(element_pos_x)))
        ap_center_elem_y = np.interp(tx_ap_center_y, element_pos_y, np.arange(len(element_pos_y)))
        # TODO Currently 'floor' NN, consider interpolating into
        #  the center delay
        ap_center_elem_x = np.floor(ap_center_elem_x).astype(np.int32)
        ap_center_elem_y = np.floor(ap_center_elem_y).astype(np.int32)
        self.tx_ap_cent_x = self.num_pkg.asarray(tx_ap_center_x).astype(np.float32)
        self.tx_ap_cent_y = self.num_pkg.asarray(tx_ap_center_y).astype(np.float32)

        # FIND THE TX_CENTER_DELAY
        # Make sure, that for all TX/RXs:
        # The center element is in the aperture
        # All TX/RX have (almost) the same delay in the aperture's center
        tx_center_delay = None
        for i, tx_rx in enumerate(seq.ops):
            tx = tx_rx.tx
            tx_center_x = ap_center_elem_x[i]
            tx_center_y = ap_center_elem_y[i]
            aperture = tx.aperture.reshape((self.n_elements, self.n_elements))
            delays = np.zeros(aperture.shape)
            delays[:] = np.nan
            delays[np.where(aperture)] = tx.delays.flatten()
            if not aperture[tx_center_y, tx_center_x]:
                # The aperture's center should transmit signal
                raise ValueError("TX aperture center should be turned on.")
            if tx_center_delay is None:
                tx_center_delay = delays[tx_center_y, tx_center_x]
            else:
                # Make sure that the center' delays is the same position for all TX/RXs
                current_center_delay = delays[tx_center_y, tx_center_x]
                if not np.isclose(tx_center_delay, current_center_delay):
                    raise ValueError(f"TX/RX {i}: center delay is not close "
                                     f"to the center delay of other TX/RXs."
                                     f"Assumed that the center element is:"
                                     f"({tx_center_y, tx_center_x}). "
                                     f"Center delays should be equalized "
                                     f"for all TX/RXs. ")

        # MIN/MAX TANG
        if self.rx_tang_limits is not None:
            self.min_tang, self.max_tang = self.rx_tang_limits
        else:
            # Default:
            self.min_tang, self.max_tang = -0.5, 0.5
        self.min_tang = self.num_pkg.float32(self.min_tang)
        self.max_tang = self.num_pkg.float32(self.max_tang)
        burst_factor = ref_tx.excitation.n_periods / (2*self.fn)
        self.initial_delay = -start_sample/65e6+burst_factor+tx_center_delay
        self.initial_delay = self.num_pkg.float32(self.initial_delay)
        self.rx_apod = scipy.signal.windows.hamming(20).astype(np.float32)
        self.rx_apod = self.num_pkg.asarray(self.rx_apod)
        self.n_rx_apod = self.num_pkg.int32(len(self.rx_apod))

        return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        data = self.num_pkg.ascontiguousarray(data)
        params = (
            self.output_buffer, data,
            self.n_seq, self.n_tx, self.n_rx_y, self.n_rx_x, self.n_samples,
            self.z_pix, self.z_size,
            self.x_pix, self.x_size,
            self.y_pix, self.y_size,
            self.sos, self.fs, self.fn,
            self.tx_foc, self.tx_ang_zx, self.tx_ang_zy,
            self.tx_ap_cent_x, self.tx_ap_cent_y,
            self.tx_ap_first_elem_x, self.tx_ap_last_elem_x,
            self.tx_ap_first_elem_y, self.tx_ap_last_elem_y,
            self.min_tang, self.max_tang,
            self.min_tang, self.max_tang,
            self.initial_delay,
            self.rx_apod, self.n_rx_apod,
            self.rx_ap_first_elem_x, self.rx_ap_first_elem_y
        )
        self._kernel(self.grid_size, self.block_size, params)
        return self.output_buffer
