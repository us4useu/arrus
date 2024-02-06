import abc
from abc import abstractmethod

import numpy as np
import math
import scipy
import scipy.signal as signal
import scipy.ndimage
import arrus.metadata
import arrus.devices.device
import arrus.devices.cpu
import arrus.devices.gpu
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
from enum import Enum
import arrus.kernels.simple_tx_rx_sequence
import arrus.kernels.tx_rx_sequence
from numbers import Number
from typing import Sequence, Dict, Callable, Union, Tuple
from arrus.params import ParameterDef, Unit, Box
from collections import defaultdict
from arrus.ops.us4r import TxRxSequence
from arrus.ops.imaging import SimpleTxRxSequence

def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None


if is_package_available("cupy"):
    import cupy
    import re

    if not re.match("^\\d+\\.\\d+\\.\\d+[a-z]*\\d*$", cupy.__version__):
        raise ValueError(f"Unrecognized pattern "
                         f"of the cupy version: {cupy.__version__}")
    m = re.search("^\\d+\\.\\d+\\.\\d+", cupy.__version__)
    if tuple(int(v) for v in m.group().split(".")) < (9, 0, 0):
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
    def __init__(self, name: str, n_elements, shape, dtype, math_pkg, type="locked"):
        if type == "locked":
            element_type = BufferElementLockBased
        elif type == "async":
            element_type = BufferElement
        else:
            raise ValueError(f"Unrecognized buffer type: {type}")
        self.input_array = [math_pkg.zeros(shape, dtype=dtype) for _ in range(n_elements)]
        self.elements = [element_type(i, data) for i, data in enumerate(self.input_array)]
        self.n_elements = n_elements
        self.name = name

    def acquire(self, pos):
        element = self.elements[pos]
        element.acquire()
        return element

    def release(self, pos):
        self.elements[pos].release()


class ProcessingRunner:
    """
    Runs processing on a specified processing device (GPU in particular).

    Currently only GPU:0 is supported.

    Currently, the input buffer should be located in CPU device,
    output buffer should be located on GPU.

    :param processings: sequence of processings
    :param metadatas: sequence of metadata objects
    """

    class State(Enum):
        READY = 1
        CLOSED = 2

    def __init__(self, input_buffer, metadatas, processings):
        import cupy as cp
        # Initialize pipeline.
        self.cp = cp
        # Pin input (host) buffer.
        self.input_buffer = self.__register_buffer(input_buffer)
        DEFAULT_BUFF = ProcessingBuffer(size=2, type="locked")

        self.in_buffers_gpu = []
        self.pipelines = []
        self.out_metadatas = []  #  sequence array, pipeline array -> metadata
        self.out_buffers = []  # sequence array, pipeline array -> buffer
        # host PC -> GPU RAM stream
        self.data_stream = cp.cuda.Stream(non_blocking=True)
        # kernel execution stream
        self.processing_stream = cp.cuda.Stream(non_blocking=True)

        self.n_arrays = len(metadatas)
        n_gpu_buffer_elements = {p.input_buffer.size if p.input_buffer is not None else DEFAULT_BUFF.size
                                 for p in processings}

        if len(n_gpu_buffer_elements) > 1:
            raise ValueError("Each GPU buffer should have exactly the same "
                             "number of elements")
        self.n_gpu_buffer_elements = next(iter(n_gpu_buffer_elements))

        # Prepare buffers and initialize pipelines.
        for metadata, processing in zip(metadatas, processings):
            in_buffer_def = processing.input_buffer
            out_buffer_def = processing.output_buffer
            in_buffer_def = in_buffer_def if in_buffer_def is not None else DEFAULT_BUFF
            out_buffer_def = out_buffer_def if out_buffer_def is not None else DEFAULT_BUFF
            gpu_buffer = Buffer(
                name=f"{metadata.context.sequence.name}/Output:0",
                n_elements=in_buffer_def.size,
                type=in_buffer_def.type,
                shape=metadata.input_shape,
                dtype=metadata.dtype,
                math_pkg=cp)
            pipeline = processing.pipeline
            out_metadata = pipeline.prepare(metadata)
            outs = [Buffer(
                name=m.name,
                n_elements=out_buffer_def.size,
                type=out_buffer_def.type,
                shape=m.input_shape,
                dtype=m.dtype,
                math_pkg=np)
                for m in out_metadata]

            self.out_metadatas.extend(out_metadata)
            self.out_buffers.append(outs)

            self.in_buffers_gpu.append(gpu_buffer)
            self.pipelines.append(pipeline)

        cp.cuda.Stream.null.synchronize()
        # Pin output buffers.
        new_out_buffers = []
        for in_array_out_buffers in self.out_buffers:
            b = self.__register_buffer(in_array_out_buffers)
            if not isinstance(b, Iterable):
                b = (b,)
            new_out_buffers.append(b)
        self.out_buffers = new_out_buffers

        # The below checks, if all processings have the callback,
        # or None of them
        is_callback = {p.callback is not None for p in processings}
        if len(is_callback) > 1:
            raise ValueError(
                "Within a single scheme, all processings should be "
                "defined using Pipelines or callbacks only - "
                "mixing these two concepts is not allowed.")
        is_callback = next(iter(is_callback))
        if is_callback:
            self.user_out_buffer = None
            cbks = [p.callback for p in processings if p.callback is not None]
            def master_callback():
                for c in overflow_callbacks:
                    c()
            self.callback = master_callback
        else:
            self.user_out_buffer = queue.Queue(maxsize=1)
            self.callback = self.default_processing_output_callback
        self._gpu_i = 0
        self._out_i = [[0]*len(ob) for ob in self.out_buffers]

        self.input_buffer.append_on_new_data_callback(self.process)

        overflow_callbacks = [p.on_buffer_overflow_callback
                              for p in processings
                              if p.on_buffer_overflow_callback is not None]
        if overflow_callbacks:
            def master_overflow_callback():
                for c in overflow_callbacks:
                    c()
            self.input_buffer.append_on_buffer_overflow_callback(
                master_overflow_callback)

        self._state = ProcessingRunner.State.READY
        self._process_lock = threading.Lock()
        self._state_lock = threading.Lock()

    @property
    def outputs(self):
        if len(self.out_metadatas) == 1:
            # Backward compatibility
            const_metadata = self.out_metadatas[0]
        else:
            const_metadata = self.out_metadatas

        if self.user_out_buffer is not None:
            return self.user_out_buffer, const_metadata
        else:
            return const_metadata

    def default_processing_output_callback(self, elements):
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
        out_elements = []
        with self._process_lock:
            for array_id, array in enumerate(input_element.arrays):
                gpu_element = self.in_buffers_gpu[array_id].acquire(self._gpu_i)
                gpu_array = gpu_element.data
                gpu_array.set(array, stream=self.data_stream)
                if array_id == self.n_arrays-1:
                    # Last array.
                    self.data_stream.launch_host_func(self.__release, input_element)
                gpu_data_ready_event = self.data_stream.record()
                self.processing_stream.wait_event(gpu_data_ready_event)
                with self.processing_stream:
                    results = self.pipelines[array_id](gpu_array)
                    # Write each result gpu array to the given output array
                    for element_id, (result, out_buffer) in enumerate(zip(results, self.out_buffers[array_id])):
                        out_i = self._out_i[array_id][element_id]
                        out_element = out_buffer.elements[out_i]
                        self._out_i[array_id][element_id] = (out_i + 1) % out_buffer.n_elements
                        out_element.acquire()
                        # TODO(ARRUS-175) Fix the issue with incomplete output data (noticed in gui4us application)
                        out_element.data[:] = result.get()
                        out_elements.append(out_element)
                self.__release(gpu_element)
            self.callback(out_elements)
            self._gpu_i = (self._gpu_i+1) % self.n_gpu_buffer_elements

    def close(self):
        with self._state_lock:
            if self._state == ProcessingRunner.State.CLOSED:
                # Already closed.
                return
            self.__unregister_buffer(self.input_buffer)
            for in_array_out_buffers in self.out_buffers:
                self.__unregister_buffer(in_array_out_buffers)
            self._state = ProcessingRunner.State.CLOSED

    def sync(self):
        self.data_stream.synchronize()
        self.processing_stream.synchronize()

    def __release(self, element):
        element.release()

    def __register_buffer(self, buffers):
        import cupy as cp
        if not isinstance(buffers, Iterable):
            buffers = (buffers,)
        for buffer in buffers:
            for element in buffer.elements:
                cp.cuda.runtime.hostRegister(element.data.ctypes.data, element.size, 1)
        if len(buffers) == 1:
            buffers = next(iter(buffers))
        return buffers

    def __unregister_buffer(self, buffers):
        import cupy as cp
        if not isinstance(buffers, Iterable):
            buffers = (buffers,)
        for buffer in buffers:
            for element in buffer.elements:
                cp.cuda.runtime.hostUnregister(element.data.ctypes.data)


class Operation:
    """
    An operation to perform in the imaging pipeline -- one data processing
    stage.

    :param name: operation name, should be unique in a given context.
        None means that a unique name should be automatically generated.
    """

    def __init__(self, name=None):
        self.name = name

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

    def set_parameter(self, key: str, value: Sequence[Number]):
        """
        Sets the value for parameter with a given name.
        :param key:
        :param value:
        :return:
        """
        raise ValueError(f"Unknown parameter: {key}")

    def get_parameter(self, key: str) -> Sequence[Number]:
        """
        Returns the current value for parameter with the given name.
        """
        raise ValueError(f"Unknown parameter: {key}")

    def get_parameters(self) -> Dict[str, ParameterDef]:
        """
        Returns description of parameters that can be set
        for this operation.
        """
        return dict()


def _get_default_op_name(op: Operation, ordinal: int):
    return f"{type(op).__name__}:{ordinal}"


def _get_op_context_param_name(op_name: str, param_name: str):
    param_name = param_name.strip()
    if not param_name.startswith("/"):
        param_name = f"/{param_name}"
    return f"/{op_name}{param_name}"


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
        return data,


class Pipeline:
    """
    Imaging pipeline.

    Processes given data using a given sequence of steps.
    The processing will be performed on a given device ('placement').
    :param steps: processing steps to run
    :param placement: device on which the processing should take place,
      default: GPU:0
    """

    def __init__(self, steps, placement=None, name=None):
        self.steps: Sequence[Operation] = steps
        self.name = name
        self._placement = None
        self._processing_stream = None
        self._input_buffer = None
        if placement is not None:
            self.set_placement(placement)
        self._set_names()
        self._param_ops: Dict[str, Tuple[Operation, str]] = {}
        self._param_defs: Dict[str, ParameterDef] = {}
        self._determine_params()

    def set_parameter(self, key: str, value: Sequence[Number]):
        """
        Sets the value for parameter with the given name.
        TODO: note: this method currently is not thread-safe
        """
        op, op_param_name = self._param_ops[key]
        op.set_parameter(op_param_name, value)

    def get_parameter(self, key: str) -> Sequence[Number]:
        """
        Returns the current value for parameter with the given name.
        """
        op, op_param_name = self._param_ops[key]
        return op.get_parameter(op_param_name)

    def get_parameters(self) -> Dict[str, ParameterDef]:
        return self._param_defs

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        outputs = deque()  # TODO avoid creating deque on each processing step
        for step in self.steps:
            if step.endpoint:
                step_outputs = step.process(data)
                # To keep the order of step_outputs, appendleft
                # collection in reversed order.
                for output in reversed(step_outputs):
                    outputs.appendleft(output)
            else:
                data = step.process(data)
        if not self._is_last_endpoint:
            outputs.appendleft(data)
        return outputs

    def __initialize(self, const_metadata):
        if not isinstance(const_metadata, Iterable):
            const_metadata = [const_metadata]
        buffers = []
        for cm in const_metadata:
            input_shape = cm.input_shape
            input_dtype = cm.dtype
            b = self.num_pkg.zeros(input_shape, dtype=input_dtype)
            b = b + 1000
            buffers.append(b)
        if len(buffers) == 1:
            # Backward compatibility
            self._input_buffer = buffers[0]
        else:
            self._input_buffer = buffers

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
                    child_metadatas = (child_metadatas,)
                # To keep the order of child_metadatas, appendleft
                # collection in reversed order.
                for metadata in reversed(child_metadatas):
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
        # Set metadata name
        for i, m in enumerate(metadatas):
            m._name = f"{self.name}/Output:{i}"
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

    def _set_names(self):
        """
        Names all the children.
        """
        type_counter = defaultdict(int)
        for step in self.steps:
            if not hasattr(step, "name") or step.name is None:
                t = type(step)
                step.name = _get_default_op_name(step, type_counter[t])
                type_counter[t] += 1

    def _determine_params(self):
        """
        Creates an internal hashmap for all parameters available
        in the pipeline.
        """
        self._param_ops = {}
        self._param_defs = {}
        for step in self.steps:
            name = step.name
            params = step.get_parameters()
            for k, param_def in params.items():
                prefixed_k = _get_op_context_param_name(name, k)
                self._param_ops[prefixed_k] = step, k
                self._param_defs[prefixed_k] = param_def


@dataclasses.dataclass(frozen=True)
class ProcessingBuffer:
    """
    :param size: the number of elements in the buffer
    :param type: buffer type ('locked')
    """
    size: int
    type: str
    # TODO: placement


class Processing:
    """
    A description of complete data processing run in the arrus.utils.imaging.
    """

    def __init__(
            self, pipeline: Pipeline,
            callback: Callable[[Sequence[Union[BufferElement, BufferElementLockBased]]], None] = None,
            input_buffer: ProcessingBuffer = None,
            output_buffer: ProcessingBuffer = None,
            on_buffer_overflow_callback=None):
        self.pipeline = pipeline
        self._pipeline_name = _get_default_op_name(self.pipeline, 0)
        self._pipeline_param_names, self._param_defs = self._determine_params()
        self.callback = callback
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.on_buffer_overflow_callback = on_buffer_overflow_callback

    def set_parameter(self, key: str, value: Sequence[Number]):
        """
        Sets the value for parameter with the given name.
        """
        pipeline_param_name = self._pipeline_param_names[key]
        self.pipeline.set_parameter(pipeline_param_name, value)

    def get_parameter(self, key: str) -> Sequence[Number]:
        """
        Returns the current value for parameter with the given name.
        """
        pipeline_param_name = self._pipeline_param_names[key]
        return self.pipeline.get_parameter(pipeline_param_name)

    def get_parameters(self) -> Dict[str, ParameterDef]:
        return self._param_defs

    def _determine_params(self):
        pipeline_param_name = {}
        param_defs = {}
        for k, param_def in self.pipeline.get_parameters().items():
            prefixed_k = _get_op_context_param_name(self._pipeline_name, k)
            pipeline_param_name[prefixed_k] = k
            param_defs[prefixed_k] = param_def
        return pipeline_param_name, param_defs


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


def _get_unique_center_frequency(sequence):
    if isinstance(sequence, arrus.ops.imaging.SimpleTxRxSequence):
        return sequence.pulse.center_frequency
    elif isinstance(sequence, arrus.ops.us4r.TxRxSequence):
        cfs = {tx_rx.tx.excitation.center_frequency for tx_rx in sequence.ops}
        if len(cfs) > 1:
            raise ValueError("Each TX/RX should have exactly the same "
                             "definition of transmit pulse.")
        return next(iter(cfs))


class BandpassFilter(Operation):
    """
    Bandpass filtering to apply to signal data.

    A bandwidth [0.5, 1.5]*center_frequency is currently used.

    The filtering is performed along the last axis.

    Currently only FIR filter is available.

    NOTE: consider using Filter op, which provides more possibilities to
    define what kind of filter is used (e.g. by providing filter coefficients).
    """

    def __init__(self, order=63, bounds=(0.5, 1.5), filter_type="hamming",
                 num_pkg=None, filter_pkg=None, **kwargs):
        """
        Bandpass filter constructor.

        Currently, the filtering is performed simply by convolving input signal
        with given type of filter. The band of frequencies is automatically
        determined basing on the TX frequency.

        :param order: filter order
        :param bounds: determines filter's frequency boundaries,
            e.g. setting 0.5 will give a bandpass filter
            [0.5*center_frequency, 1.5*center_frequency].
        :param filter_type: one of "butter" (for Butterworth coefficients)
            or one of windows provided by scipy.signal.get_window
        """
        self.taps = None
        self.order = order
        self.bound_l, self.bound_r = bounds
        self.filter_type = filter_type
        self.xp = num_pkg
        self.filter_pkg = filter_pkg
        self.kwargs = kwargs

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        l, r = self.bound_l, self.bound_r
        center_frequency = _get_unique_center_frequency(const_metadata.context.sequence)
        sampling_frequency = const_metadata.data_description.sampling_frequency
        band = [l * center_frequency, r * center_frequency]
        if self.filter_type == "butter":
            taps, _ = scipy.signal.butter(
                self.order, band,
                btype='bandpass', fs=sampling_frequency)
        else:
            taps = scipy.signal.firwin(
                numtaps=self.order,
                cutoff=band,
                pass_zero=False,
                window=self.filter_type,
                fs=sampling_frequency,
                **self.kwargs
            )
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
        fc = _get_unique_center_frequency(const_metadata.context.sequence)
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


class DigitalDownConversion(Operation):
    """
    IQ demodulation, decimation.
    """

    def __init__(self, decimation_factor, fir_params=None,
                 fir_cutoff_relative=1.0, fir_order=15, fir_type="hamming"):
        self.decimation_factor = decimation_factor
        self.fir_cutoff_relative = fir_cutoff_relative
        self.fir_order = fir_order
        self.fir_type = fir_type
        self.fir_params = fir_params if fir_params is not None else {}

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def prepare(self, const_metadata):
        self.demodulator = QuadratureDemodulation()
        center_frequency = _get_unique_center_frequency(const_metadata.context.sequence)
        sampling_frequency = const_metadata.data_description.sampling_frequency
        cutoff_freq = center_frequency * self.fir_cutoff_relative
        fir_coefficients = scipy.signal.firwin(
            numtaps=self.fir_order,
            cutoff=cutoff_freq,
            window=self.fir_type,
            fs=sampling_frequency,
            pass_zero="lowpass",
            **self.fir_params
        )
        self.decimator = Decimation(
            self.decimation_factor,
            filter_coeffs=fir_coefficients, filter_type="fir")
        self.demodulator.set_pkgs(num_pkg=self.xp, filter_pkg=self.filter_pkg)
        self.decimator.set_pkgs(num_pkg=self.xp, filter_pkg=self.filter_pkg)
        const_metadata = self.demodulator.prepare(const_metadata)
        return self.decimator.prepare(const_metadata)

    def process(self, data):
        data = self.demodulator.process(data)
        return self.decimator.process(data)


class Decimation(Operation):
    """
    Downsampling + Low-pass filter.

    By default CIC filter is used.
    """

    def __init__(self, decimation_factor, filter_type="cic",
                 filter_coeffs=None, cic_order=2, num_pkg=None):
        """
        Decimation.

        :param decimation_factor: decimation factor to apply
        """
        self.decimation_factor = decimation_factor
        self.xp = num_pkg
        if filter_type == "cic":
            self.filter_coeffs = self._get_cic_filter_coeffs(
                decimation_factor=self.decimation_factor,
                order=cic_order
            )
        elif filter_type == "fir":
            if filter_coeffs is None:
                raise ValueError("Decimation: FIR filter requires "
                                 "manually specified filter coefficients.")
            self.filter_coeffs = filter_coeffs
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def _get_cic_filter_coeffs(self, decimation_factor, order):
        cicFir = np.array([1], dtype=np.float32)
        cicFir1 = np.ones(decimation_factor, dtype=np.float32)
        for _ in range(order):
            cicFir = np.convolve(cicFir, cicFir1, 'full')
        return cicFir

    def prepare(self, const_metadata):
        new_fs = (const_metadata.data_description.sampling_frequency
                  / self.decimation_factor)
        new_signal_description = dataclasses.replace(
            const_metadata.data_description,
            sampling_frequency=new_fs)
        input_shape = const_metadata.input_shape
        n_samples = input_shape[-1]
        self.filter_coeffs = self.xp.asarray(self.filter_coeffs)
        self.filter_coeffs = self.filter_coeffs.astype(self.xp.float32)
        output_shape = input_shape[:-1] + (math.ceil(n_samples / self.decimation_factor),)
        return const_metadata.copy(data_desc=new_signal_description,
                                   input_shape=output_shape)

    def process(self, data):
        fir_output = self.filter_pkg.convolve1d(
            data, self.filter_coeffs, axis=-1, mode='constant')
        data_out = fir_output[..., 0::self.decimation_factor]
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
        seq = const_metadata.context.sequence
        probe_model = get_unique_probe_model(const_metadata)
        if probe_model.is_convex_array():
            raise ValueError("Phased array scanning is implemented for "
                             "linear phased arrays only.")

        self._kernel_module = _read_kernel_module("rx_beamforming.cu")
        self._kernel = self._kernel_module.get_function("beamformPhasedArray")

        self.n_seq, self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.output_buffer = cp.zeros((self.n_seq, self.n_tx, self.n_samples), dtype=cp.complex64)

        self.tx_angles = cp.asarray(seq.angles, dtype=cp.float32)

        device_fs = const_metadata.context.device.sampling_frequency
        acq_fs = (device_fs / seq.downsampling_factor)
        fs = const_metadata.data_description.sampling_frequency
        fc = seq.pulse.center_frequency
        n_periods = seq.pulse.n_periods
        medium = const_metadata.context.medium
        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound

        rx_sample_range = arrus.kernels.simple_tx_rx_sequence.get_sample_range(
            op=seq, fs=fs, speed_of_sound=c)
        start_sample, end_sample = rx_sample_range
        initial_delay = - start_sample / acq_fs
        if seq.init_delay == "tx_start":
            burst_factor = n_periods / (2 * fc)
            tx_center_delay = arrus.kernels.simple_tx_rx_sequence.get_center_delay(
                sequence=seq, c=c, probe_model=probe_model, fs=fs)
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
        self.start_time = cp.float32(start_sample / acq_fs)
        self.init_delay = cp.float32(initial_delay)
        self.max_tang = cp.float32(max_tang)
        sample_block_size = min(self.n_samples, 16)
        scanline_block_size = min(self.n_tx, 16)
        n_seq_block_size = min(self.n_seq, 4)
        self.block_size = (sample_block_size, scanline_block_size, n_seq_block_size)
        self.grid_size = (int((self.n_samples - 1) // sample_block_size + 1),
                          int((self.n_tx - 1) // scanline_block_size + 1),
                          int((self.n_seq - 1) // n_seq_block_size + 1))
        # xElemConst
        # Get aperture origin (for the given aperture center element/aperture center)
        tx_rx_params = arrus.kernels.simple_tx_rx_sequence.preprocess_sequence_parameters(
            probe_model, seq)
        # There is a single TX and RX aperture center for all TX/RXs
        rx_aperture_center_element = np.array(tx_rx_params["rx_ap_cent"])[0]
        rx_aperture_origin = _get_rx_aperture_origin(
            rx_aperture_center_element, seq.rx_aperture_size)
        rx_aperture_offset = rx_aperture_center_element - rx_aperture_origin
        x_elem = (np.arange(0, self.n_rx) - rx_aperture_offset) * probe_model.pitch
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
        seq = const_metadata.context.sequence
        probe_model = get_unique_probe_model(const_metadata)
        raw_seq = const_metadata.context.raw_sequence
        medium = const_metadata.context.medium
        tx_rx_params = arrus.kernels.simple_tx_rx_sequence.preprocess_sequence_parameters(probe_model, seq)
        rx_aperture_center_element = np.array(tx_rx_params["tx_ap_cent"])

        self.n_seq, self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.is_iq = const_metadata.is_iq_data
        if self.is_iq:
            buffer_dtype = self.xp.complex64
        else:
            buffer_dtype = self.xp.float32

        # -- Output buffer
        self.buffer = self.xp.zeros(
            (self.n_seq * self.n_tx, self.n_rx * self.n_samples),
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
        rx_sample_range = arrus.kernels.simple_tx_rx_sequence.get_sample_range(
            op=seq, fs=fs, speed_of_sound=c)
        start_sample = rx_sample_range[0]
        rx_aperture_origin = _get_rx_aperture_origin(rx_aperture_center_element, seq.rx_aperture_size)
        # -start_sample compensates the fact, that the data indices always
        # start from 0
        initial_delay = - start_sample / acq_fs
        if seq.init_delay == "tx_start":
            burst_factor = n_periods / (2 * fc)
            tx_center_delay = arrus.kernels.simple_tx_rx_sequence.get_center_delay(
                sequence=seq, c=c, probe_model=probe_model, fs=fs)
            initial_delay += tx_center_delay + burst_factor
        elif not seq.init_delay == "tx_center":
            raise ValueError(f"Unrecognized init_delay value: {initial_delay}")
        radial_distance = (
                (start_sample / acq_fs + np.arange(0, self.n_samples) / fs) * c / 2
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
        self.delays[np.isclose(self.delays, self.n_samples - 1)] = self.n_samples - 1
        self.delays[self.delays > self.n_samples - 1] = total_n_samples + 1
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
        rx_apodization = rx_apodization / (rx_apod_sum.reshape(1, self.n_samples))
        self.rx_apodization = self.xp.asarray(rx_apodization)
        # IQ correction
        self.t = self.xp.asarray(self.t)
        self.iq_correction = self.xp.exp(1j * 2 * np.pi * fc * self.t) \
            .astype(self.xp.complex64)
        # Create new output shape
        return const_metadata.copy(input_shape=(self.n_seq, self.n_tx, self.n_samples))

    def process(self, data):
        data = data.copy().reshape(self.n_seq * self.n_tx, self.n_rx * self.n_samples)

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
        super().__init__()
        self.axes = axes
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        input_spacing = const_metadata.data_description.spacing
        axes = list(range(len(input_shape)))[::-1] if self.axes is None else self.axes
        output_shape = tuple(input_shape[ax] for ax in axes)
        if input_spacing is not None:
            output_spacing = tuple(input_spacing.coordinates[ax] for ax in axes)
            new_signal_description = dataclasses.replace(
                const_metadata.data_description,
                spacing=arrus.metadata.Grid(
                    coordinates=output_spacing
                )
            )
            return const_metadata.copy(
                input_shape=output_shape,
                data_desc=new_signal_description
            )
        else:
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
        probe = get_unique_probe_model(const_metadata)

        new_signal_description = dataclasses.replace(
            const_metadata.data_description,
            spacing=arrus.metadata.Grid(
                coordinates=(self.z_grid, self.x_grid)
            )
        )
        const_metadata = const_metadata.copy(
            data_desc=new_signal_description
        )
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
        probe = get_unique_probe_model(const_metadata)
        tx_rx_params = arrus.kernels.simple_tx_rx_sequence.preprocess_sequence_parameters(probe, seq)
        tx_aperture_center_element = tx_rx_params["tx_ap_cent"]
        n_elements = probe.n_elements
        if n_elements % 2 != 0:
            raise ValueError("Even number of probe elements is required.")
        pitch = probe.pitch
        data_desc = const_metadata.data_description
        c = _get_speed_of_sound(const_metadata.context)
        tx_center_diff = np.diff(tx_aperture_center_element)
        # Check if tx aperture centers are evenly spaced.
        if not np.allclose(tx_center_diff, [tx_center_diff[0]] * len(tx_center_diff)):
            raise ValueError("Transmits should be done by consecutive "
                             "center elements (got tx center elements: "
                             f"{tx_aperture_center_element}")
        tx_center_diff = tx_center_diff[0]
        # Determine input grid.
        input_x_grid_diff = tx_center_diff * pitch
        input_x_grid_origin = (tx_aperture_center_element[0] - (n_elements - 1) / 2) * pitch
        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = data_desc.sampling_frequency
        rx_sample_range = arrus.kernels.simple_tx_rx_sequence.get_sample_range(
            op=seq, fs=fs, speed_of_sound=c)
        start_sample = rx_sample_range[0]
        input_z_grid_origin = start_sample / acq_fs * c / 2
        input_z_grid_diff = c / (fs * 2)
        # Map x_grid and z_grid to the RF frame coordinates.
        interp_x_grid = (self.x_grid - input_x_grid_origin) / input_x_grid_diff
        interp_z_grid = (self.z_grid - input_z_grid_origin) / input_z_grid_diff
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
        probe = get_unique_probe_model(const_metadata)
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

        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound

        rx_sample_range = arrus.kernels.simple_tx_rx_sequence.get_sample_range(
            op=seq, fs=fs, speed_of_sound=c)
        start_sample = rx_sample_range[0]

        tx_ap_cent_ang, _, _ = arrus.kernels.tx_rx_sequence.get_aperture_center(
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
            if not np.allclose(diffs, [diffs[0]] * len(diffs)):
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
        probe = get_unique_probe_model(const_metadata)
        data_desc = const_metadata.data_description

        self.n_frames, n_samples, n_scanlines = const_metadata.input_shape
        seq = const_metadata.context.sequence
        fs = const_metadata.context.device.sampling_frequency
        acq_fs = fs / seq.downsampling_factor
        fs = data_desc.sampling_frequency
        c = _get_speed_of_sound(const_metadata.context)

        rx_sample_range = arrus.kernels.simple_tx_rx_sequence.get_sample_range(
            op=seq, fs=fs, speed_of_sound=c)

        start_sample, _ = rx_sample_range
        start_time = start_sample / acq_fs
        tx_rx_params = arrus.kernels.simple_tx_rx_sequence.preprocess_sequence_parameters(probe, seq)
        tx_ap_cent_elem = np.array(tx_rx_params["tx_ap_cent"])[0]
        tx_ap_cent_ang, tx_ap_cent_x, tx_ap_cent_z = arrus.kernels.tx_rx_sequence.get_aperture_center(
            tx_ap_cent_elem, probe)

        # There is a single position of TX aperture.
        tx_ap_cent_x = tx_ap_cent_x.squeeze().item()
        tx_ap_cent_z = tx_ap_cent_z.squeeze().item()
        tx_ap_cent_ang = tx_ap_cent_ang.squeeze().item()

        self.radGridIn = (start_time + np.arange(0, n_samples) / fs) * c / 2
        self.azimuthGridIn = seq.angles + tx_ap_cent_ang
        azimuthGridOut = np.arctan2((self.x_grid - tx_ap_cent_x), (self.z_grid.T - tx_ap_cent_z))
        radGridOut = np.sqrt((self.x_grid - tx_ap_cent_x) ** 2 + (self.z_grid.T - tx_ap_cent_z) ** 2)
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
        return 20 * self.num_pkg.log10(data)


class DynamicRangeAdjustment(Operation):
    """
    Clips data values to given range.
    """

    def __init__(self, min=20, max=80, name=None):
        """
        Constructor.

        :param min: minimum value to clamp
        :param max: maximum value to clamp
        """
        super().__init__(name=name)
        self.min = min
        self.max = max
        self.xp = None

    def set_parameter(self, key: str, value: Sequence[Number]):
        if not hasattr(self, key):
            raise ValueError(f"{type(self).__name__} has no {key} parameter.")
        setattr(self, key, value)

    def get_parameter(self, key: str) -> Sequence[Number]:
        if not hasattr(self, key):
            raise ValueError(f"{type(self).__name__} has no {key} parameter.")
        return getattr(self, key)

    def get_parameters(self) -> Dict[str, ParameterDef]:
        return {
            "min": ParameterDef(
                name="min",
                space=Box(
                    shape=(1, ),
                    dtype=np.float32,
                    unit=Unit.dB,
                    low=-np.inf,
                    high=np.inf
                ),
            ),
            "max": ParameterDef(
                name="max",
                space=Box(
                    shape=(1, ),
                    dtype=np.float32,
                    unit=Unit.dB,
                    low=-np.inf,
                    high=np.inf
                ),
            )
        }

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
        return const_metadata.copy(dtype=np.uint8)

    def process(self, data):
        data = data - self.xp.min(data)
        data = data / self.xp.max(data) * 255
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
            n_frames = self.num_pkg.max(fcm_frames[fcm_us4oems == us4oem]) + 1
            us4oem_offset = fcm.frame_offsets[us4oem]
            # NOTE: below we use only a single sequence
            src_start = us4oem_offset * n_samples + self.sequence[0] * n_frames * n_samples
            src_end = src_start + n_frames * n_samples
            dst_end = dst_start + n_frames * n_samples
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
        output_shape = (n_seq,) + output_shape
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
        super().__init__()
        if isinstance(frames, np.ndarray):
            frames = frames.tolist()
        self.frames = tuple(frames)

    def set_pkgs(self, **kwargs):
        pass

    def prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        context = const_metadata.context
        seq = context.sequence
        n_frames = len(self.frames)

        if len(input_shape) == 3:
            input_n_frames, d2, d3 = input_shape
            output_shape = n_frames, d2, d3
            self.selector = [slice(None)]*3
            self.selector[0] = self.frames
        elif len(input_shape) == 4:
            n_seq, input_n_frames, d2, d3 = input_shape
            output_shape = n_seq, n_frames, d2, d3
            self.selector = [slice(None)]*4
            self.selector[1] = self.frames
        else:
            raise ValueError("The input should be 3-D or 4-D "
                             "(frame number should be the first or second axis)")

        self.selector = tuple(self.selector)

        # Adapt sequence and raw sequence to the changes in the number of
        # frames.
        new_raw_ops = self._limit_list(
            const_metadata.context.raw_sequence.ops,
            self.frames
        )
        new_raw_seq = dataclasses.replace(
            const_metadata.context.raw_sequence,
            ops=new_raw_ops
        )
        if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
            # select appropriate angles
            angles = self._limit_params(seq.angles, self.frames)
            tx_focus = self._limit_params(seq.tx_focus, self.frames)
            tx_aperture_center_element = self._limit_params(
                seq.tx_aperture_center_element, self.frames)
            tx_aperture_center = self._limit_params(
                seq.tx_aperture_center, self.frames)
            rx_aperture_center_element = self._limit_params(
                seq.rx_aperture_center_element, self.frames)
            rx_aperture_center = self._limit_params(
                seq.rx_aperture_center, self.frames)

            new_seq = dataclasses.replace(
                seq,
                angles=angles,
                tx_focus=tx_focus,
                tx_aperture_center_element=tx_aperture_center_element,
                tx_aperture_center=tx_aperture_center,
                rx_aperture_center_element=rx_aperture_center_element,
                rx_aperture_center=rx_aperture_center)
            new_context = dataclasses.replace(
                const_metadata.context,
                sequence=new_seq,
                raw_sequence=new_raw_seq)
            return const_metadata.copy(input_shape=output_shape,
                                       context=new_context)
        elif isinstance(seq, arrus.ops.us4r.TxRxSequence):
            new_ops = self._limit_list(
                const_metadata.context.sequence.ops,
                self.frames
            )
            new_seq = dataclasses.replace(
                const_metadata.context.sequence,
                ops=new_ops
            )
            new_context = dataclasses.replace(
                const_metadata.context,
                sequence=new_seq,
                raw_sequence=new_raw_seq)
            return const_metadata.copy(input_shape=output_shape,
                                       context=new_context)
        else:
            return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        return data[self.selector]

    def _limit_params(self, value, frames):
        if value is not None and hasattr(value, "__len__") and len(value) > 1:
            return np.array(value)[frames]
        else:
            return value

    def _limit_list(self, l, frames):
        if l is not None:
            return [e for i, e in enumerate(l) if i in frames]


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
        super().__init__()
        self.x_grid = x_grid
        self.z_grid = z_grid
        import cupy as cp
        self.num_pkg = cp
        self.rx_tang_limits = rx_tang_limits  # Currently used only by Convex PWI implementation

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
        self.fs = self.num_pkg.float32(const_metadata.data_description.sampling_frequency)
        probe_model = get_unique_probe_model(const_metadata)

        if isinstance(seq, SimpleTxRxSequence):
            rx_op = seq
            tx_op = seq
            rx_sample_range = arrus.kernels.simple_tx_rx_sequence.get_sample_range(
                op=seq, fs=self.fs, speed_of_sound=seq.speed_of_sound)
            angles = np.atleast_1d(np.asarray(seq.angles))
            focus = np.atleast_1d(np.asarray(seq.tx_focus))
            #
            # TX aperture description
            # Convert the sequence to the positions of the aperture centers
            tx_rx_params = arrus.kernels.simple_tx_rx_sequence.compute_tx_rx_params(
                probe_model, seq)
            tx_centers, tx_sizes = tx_rx_params["tx_ap_cent"], tx_rx_params["tx_ap_size"]
            rx_centers, rx_sizes = tx_rx_params["rx_ap_cent"], tx_rx_params["rx_ap_size"]

            tx_center_delay = arrus.kernels.simple_tx_rx_sequence.get_center_delay(
                sequence=seq, c=seq.speed_of_sound, probe_model=probe_model,
                fs=const_metadata.data_description.sampling_frequency
            )
        elif isinstance(seq, TxRxSequence):
            self._assert_unique(seq, lambda op: op.tx.excitation.center_frequency, "center_frequency")
            self._assert_unique(seq, lambda op: op.tx.excitation.n_periods, "n_periods")
            self._assert_unique(seq, lambda op: op.rx.downsampling_factor, "downsampling_factor")
            self._assert_unique(seq, lambda op: op.rx.sample_range, "sample_range")
            self._assert_unique(seq, lambda op: op.tx.speed_of_sound, "speed_of_sound")
            # Reference TX/RX ops
            rx_op = seq.ops[0].rx
            tx_op = seq.ops[0].tx
            rx_sample_range = rx_op.sample_range
            angles = np.asarray([op.tx.angle for op in seq.ops])
            focus = np.asarray([op.tx.focus for op in seq.ops])
            if tx_op.angle is None and tx_op.focus is None:
                raise ValueError("It is required to provide sequence with "
                                 "transmit angles in order to run "
                                 "the ReconstructLri operator.")
            tx_apertures = [op.tx.aperture for op in seq.ops]
            rx_apertures = [op.rx.aperture for op in seq.ops]
            tx_centers = arrus.kernels.tx_rx_sequence.get_apertures_center_elements(
                apertures=tx_apertures, probe_model=probe_model
            )
            tx_sizes = arrus.kernels.tx_rx_sequence.get_apertures_sizes(
                apertures=tx_apertures, probe_model=probe_model
            )
            rx_centers = arrus.kernels.tx_rx_sequence.get_apertures_center_elements(
                apertures=rx_apertures, probe_model=probe_model
            )
            rx_sizes = arrus.kernels.tx_rx_sequence.get_apertures_sizes(
                apertures=rx_apertures, probe_model=probe_model
            )
            tx_center_delay = arrus.kernels.tx_rx_sequence.get_center_delay(
                sequence=seq, probe_model=probe_model
            )
        else:
            raise ValueError(f"Unsupported type of sequence: {seq}")

        if len(focus) == 1:
            focus = np.repeat(focus, self.n_tx)
        if len(angles) == 1:
            angles = np.repeat(angles, self.n_tx)

        # tx center value is calculated with the assumption that the delays
        # are "normalized to 0 values (i.e. min(all_delays) == 0.
        # We need to take into account the actual delays here:
        # the min(delays) may be != 0, e.g. when we reconstructing only some
        # subsequence of the original sequence.
        tx_center_delay -= self._get_min_delay(const_metadata.context.raw_sequence)

        self.x_size = len(self.x_grid)
        self.z_size = len(self.z_grid)
        output_shape = (self.n_seq, self.n_tx, self.x_size, self.z_size)
        self.output_buffer = self.num_pkg.zeros(output_shape, dtype=self.num_pkg.complex64)
        x_block_size = min(self.x_size, 16)
        z_block_size = min(self.z_size, 16)
        tx_block_size = min(self.n_tx, 4)
        self.block_size = (z_block_size, x_block_size, tx_block_size)
        self.grid_size = (int((self.z_size - 1) // z_block_size + 1),
                          int((self.x_size - 1) // x_block_size + 1),
                          int((self.n_seq * self.n_tx - 1) // tx_block_size + 1))
        self.x_pix = self.num_pkg.asarray(self.x_grid, dtype=self.num_pkg.float32)
        self.z_pix = self.num_pkg.asarray(self.z_grid, dtype=self.num_pkg.float32)

        # System and transmit properties.
        self.sos = self.num_pkg.float32(tx_op.speed_of_sound)
        self.fn = self.num_pkg.float32(tx_op.excitation.center_frequency)
        self.pitch = self.num_pkg.float32(probe_model.pitch)
        start_sample = rx_sample_range[0]

        # Probe description
        element_pos_x = probe_model.element_pos_x
        element_pos_z = probe_model.element_pos_z
        element_angle_tang = np.tan(probe_model.element_angle)
        self.n_elements = probe_model.n_elements

        device_props = cp.cuda.runtime.getDeviceProperties(0)
        if device_props["totalConstMem"] < 256 * 3 * 4:  # 3 float32 arrays, 256 elements max
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

        tx_center_angles, tx_center_x, tx_center_z = arrus.kernels.tx_rx_sequence.get_aperture_center(
            tx_centers, probe_model)
        tx_center_angles = tx_center_angles + angles
        self.tx_ang_zx = self.num_pkg.asarray(tx_center_angles, dtype=self.num_pkg.float32)
        self.tx_ap_cent_x = self.num_pkg.asarray(tx_center_x, dtype=self.num_pkg.float32)
        self.tx_ap_cent_z = self.num_pkg.asarray(tx_center_z, dtype=self.num_pkg.float32)

        # first/last probe element in TX aperture
        tx_ap_origin = np.round(tx_centers - (tx_sizes - 1) / 2 + 1e-9).astype(np.int32)
        rx_ap_origin = np.round(rx_centers - (rx_sizes - 1) / 2 + 1e-9).astype(np.int32)
        tx_ap_first_elem = np.maximum(tx_ap_origin, 0)
        tx_ap_last_elem = np.minimum(tx_ap_origin + tx_sizes - 1, probe_model.n_elements - 1)
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

        self.tx_foc = self.num_pkg.asarray(focus, dtype=self.num_pkg.float32)
        burst_factor = tx_op.excitation.n_periods/(2 * self.fn)
        self.initial_delay = -start_sample/65e6+burst_factor+tx_center_delay
        self.initial_delay = self.num_pkg.float32(self.initial_delay)
        # Output metadata
        new_signal_description = dataclasses.replace(
            const_metadata.data_description,
            spacing=arrus.metadata.Grid(
                coordinates=(self.x_grid, self.z_grid)
            )
        )
        return const_metadata.copy(
            input_shape=output_shape,
            data_desc=new_signal_description
        )

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

    def _assert_unique(self, seq: TxRxSequence, getter: Callable, name: str):
        s = {getter(op) for op in seq.ops}
        if len(s) > 1:
            raise ValueError("The following property should be unique "
                             f"in the sequence: {name}, found values: "
                             f"{s}")

    def _get_min_delay(self, raw_sequence):
        all_delays = [np.min(op.tx.delays) for op in raw_sequence.ops]
        return np.min(all_delays)


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
        actual_axis = len(output_shape) - 1 if self.axis == -1 else self.axis
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
        actual_axis = len(output_shape) - 1 if self.axis == -1 else self.axis
        del output_shape[actual_axis]
        return const_metadata.copy(input_shape=tuple(output_shape))

    def process(self, data):
        return self.num_pkg.mean(data, axis=self.axis)


def _get_rx_aperture_origin(aperture_center_element, aperture_size):
    return np.round(aperture_center_element - (aperture_size - 1) / 2 + 1e-9)


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
                    or src_channel != prev_src_channel + 1):
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
            from arrus.utils.us4r_remap_gpu import get_default_grid_block_size, run_remap_v1
            self._fcm_frames = cp.asarray(fcm.frames)
            self._fcm_channels = cp.asarray(fcm.channels)
            self._fcm_us4oems = cp.asarray(fcm.us4oems)
            frame_offsets = fcm.frame_offsets
            #  TODO constant memory
            self._frame_offsets = cp.asarray(frame_offsets)
            # For each us4OEM, get number of physical frames this us4OEM gathers.
            # Note: this is the max number of us4OEMs IN USE.
            n_us4oems = cp.max(self._fcm_us4oems).get() + 1
            n_frames_us4oems = []
            # The us4OEM:0 is a master us4OEM that collects data from all transmits,
            # even if its channels are not included in the RX aperture (each frame
            # contains frame metadata information).
            n_frames_us4oems.append(fcm.n_frames[0]//batch_size)
            for us4oem in range(1, n_us4oems):
                us4oem_frames = self._fcm_frames[self._fcm_us4oems == us4oem]
                if us4oem_frames.size == 0:
                    n_frames_us4oems.append(0)
                else:
                    n_frames_us4oem = cp.max(us4oem_frames).get().item() + 1
                    n_frames_us4oems.append(n_frames_us4oem)

            #  TODO constant memory
            self._n_frames_us4oems = cp.asarray(n_frames_us4oems, dtype=cp.uint32)
            self.grid_size, self.block_size = get_default_grid_block_size(
                self._fcm_frames, n_samples,
                batch_size
            )

            def gpu_remap_fn(data):
                run_remap_v1(self.grid_size, self.block_size,
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


class RemapToLogicalOrderV2(Operation):
    """
    Remaps the order of the data to logical order defined by the us4r device.

    If the batch size was equal 1, the raw ultrasound RF data with shape.
    (1, n_frames, n_channels, n_samples, n_components).
    A single metadata object will be returned.

    If the batch size was > 1, the raw ultrasound RF data with shape
    (n_us4oems*n_samples*n_frames*n_batches, n_components, 32) will be reordered
    to (batch_size, n_frames, n_channels, n_samples, n_components).
    A list of metadata objects will be returned.
    """

    def __init__(self, num_pkg=None):
        self._output_buffer = None
        self.xp = num_pkg
        self.remap = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _is_prepared(self):
        return self._output_buffer is not None

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

        input_order = len(const_metadata.input_shape)
        # Input: RF data: (total_n_samples, 32),
        # IQ data: (total_n_samples, 2, 32)
        n_components = 1 if input_order == 2 else 2
        self.output_shape = (batch_size, n_frames, n_channels, n_samples, n_components)
        self._output_buffer = xp.zeros(shape=self.output_shape, dtype=xp.int16)
        if xp == np:
            # CPU
            raise ValueError(f"'{type(self).__name__}' is not implemented for CPU")
        else:
            # GPU
            import cupy as cp
            from arrus.utils.us4r_remap_gpu import get_default_grid_block_size_v2, run_remap_v2
            self._fcm_frames = cp.asarray(fcm.frames)
            self._fcm_channels = cp.asarray(fcm.channels)
            self._fcm_us4oems = cp.asarray(fcm.us4oems)
            frame_offsets = fcm.frame_offsets
            #  TODO constant memory
            self._frame_offsets = cp.asarray(frame_offsets)
            # For each us4OEM, get number of physical frames this us4OEM gathers.
            # Note: this is the maximum id of us4OEM IN USE.
            n_us4oems = cp.max(self._fcm_us4oems).get() + 1
            n_frames_us4oems = []
            for us4oem in range(n_us4oems):
                us4oem_frames = self._fcm_frames[self._fcm_us4oems == us4oem]
                if us4oem_frames.size == 0:
                    n_frames_us4oems.append(0)
                else:
                    n_frames_us4oem = cp.max(us4oem_frames).get().item()
                    n_frames_us4oems.append(n_frames_us4oem)
            #  TODO constant memory
            self._n_frames_us4oems = cp.asarray(n_frames_us4oems, dtype=cp.uint32) + 1
            self.grid_size, self.block_size = get_default_grid_block_size_v2(
                self._fcm_frames, n_samples,
                batch_size
            )

            def gpu_remap_fn(data):
                run_remap_v2(self.grid_size, self.block_size,
                             [self._output_buffer, data,
                              self._fcm_frames, self._fcm_channels,
                              self._fcm_us4oems, self._frame_offsets,
                              self._n_frames_us4oems,
                              batch_size, n_frames, n_samples, n_channels,
                              n_components])

            self._remap_fn = gpu_remap_fn
        return const_metadata.copy(input_shape=self.output_shape)

    def process(self, data):
        self._remap_fn(data)
        return self._output_buffer


class ToRealOrComplex(Operation):
    """
    Converts the input array with shape
    (..., n_components), a particular dtype, to:
    - complex64 with shape (...) if n_components == 2 (regardles of dtype)
    - dtype with shape (...) if n_components == 1 (simply, removes the last
      axis).
    """

    def __init__(self, num_pkg=None):
        self._output_buffer = None
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        input_shape = const_metadata.input_shape
        n_components = input_shape[-1]

        output_shape = input_shape[:-1]
        if n_components == 2:
            output_dtype = self.xp.complex64
            self.process = self._process_to_complex
        elif n_components == 1:
            output_dtype = const_metadata.dtype
            self.process = self._process_to_real
        else:
            raise ValueError(f"Unhandled number of components: {n_components},"
                             f"should be 1 (real) or 2 (complex).")

        # get shape, create an array with given shape
        # create required transfers
        return const_metadata.copy(input_shape=output_shape,
                                   dtype=output_dtype)

    def _process_to_complex(self, data):
        output = data.astype(self.xp.float32)
        return output[..., 0] + 1j * output[..., 1]

    def _process_to_real(self, data):
        return data[..., 0]


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

        input_shape = const_metadata.input_shape
        is_ddc = len(input_shape) == 3
        self._slices = (slice(0, self._n_samples * self._n_frames, self._n_samples),)
        if is_ddc:
            self._slices = self._slices + (0,)  # Select "I" value.
        return const_metadata

    def process(self, data):
        return data[self._slices] \
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
        probe_model = get_unique_probe_model(const_metadata)
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
        self.grid_size = (int((self.z_size - 1) // z_block_size + 1),
                          int((self.x_size - 1) // x_block_size + 1),
                          int((self.y_size - 1) // y_block_size + 1))

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
        pitch = probe_model.pitch
        self.n_elements = 32
        n_rows_x = self.n_elements
        n_rows_y = self.n_elements + 3
        # General regular position of elements
        element_pos_x = np.linspace(-(n_rows_x - 1) / 2, (n_rows_x - 1) / 2, num=n_rows_x)
        element_pos_x = element_pos_x * pitch

        element_pos_y = np.linspace(-(n_rows_y - 1) / 2, (n_rows_y - 1) / 2, num=n_rows_y)
        element_pos_y = element_pos_y * pitch
        element_pos_y = np.delete(element_pos_y, (8, 17, 26))

        element_pos_x = element_pos_x.astype(np.float32)
        element_pos_y = element_pos_y.astype(np.float32)
        # Put the data into GPU constant memory.
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        if device_props["totalConstMem"] < 256 * 2 * 4:  # 2 float32 arrays, 256 elements max
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
        rxap_size_x = set((rxap_max_x - rxap_min_x).tolist())
        rxap_size_y = set((rxap_max_y - rxap_min_y).tolist())
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
        tx_ap_center_x = (element_pos_x[txap_min_x] + element_pos_x[txap_max_x]) / 2
        tx_ap_center_y = (element_pos_y[txap_min_y] + element_pos_y[txap_max_y]) / 2
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
        burst_factor = ref_tx.excitation.n_periods / (2 * self.fn)
        self.initial_delay = -start_sample / 65e6 + burst_factor + tx_center_delay
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


class Equalize(Operation):
    """
    Equalize means values along a specific axis.
    """

    def __init__(self, axis=0, axis_offset=0, num_pkg=None):
        self.axis = axis
        self.axis_offset = axis_offset
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        self.input_dtype = const_metadata.dtype
        self.input_shape = const_metadata.input_shape
        if self.axis >= len(self.input_shape):
            raise ValueError(f"Equalize: axis out of bounds: {self.axis}, "
                             f"for shape: {self.input_shape}.")
        self.slice = [slice(None)] * len(self.input_shape)
        self.slice[self.axis] = slice(self.axis_offset, None)
        self.slice = tuple(self.slice)
        return const_metadata.copy()

    def process(self, data):
        d = data[self.slice]
        m = d.mean(axis=self.axis).astype(self.input_dtype)
        m = self.xp.expand_dims(m, axis=self.axis)
        return data - m


class DelayAndSumLUT(Operation):
    """
    Delay and sum using look-up tables.

    This operator requires GPU and cupy package installed.
    TODO Note:: the below operator will not work correctly for:
    - start_sample != 0,
    - downsampling_factor != 1.
    """

    def __init__(self,
                 tx_delays, tx_apodization,
                 rx_apodization, rx_delays,
                 output_type=None):
        self.tx_delays = tx_delays
        self.rx_delays = rx_delays
        self.tx_apodization = tx_apodization
        self.rx_apodization = rx_apodization
        import cupy as cp
        self.num_pkg = cp
        self.output_type = output_type if output_type is not None else "hri"

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg is np:
            raise ValueError("ReconstructLri operation is implemented for GPU only.")

    def prepare(self, const_metadata):
        import cupy as cp
        current_dir = os.path.dirname(os.path.join(os.path.abspath(__file__)))
        _kernel_source = Path(os.path.join(current_dir, "das_lut.cu")).read_text()
        self._kernel_module = self.num_pkg.RawModule(code=_kernel_source)

        # INPUT PARAMETERS.
        # Input data shape.
        self.n_seq, self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.n_tx, self.y_size, self.x_size, self.z_size = self.tx_delays.shape
        self.n_rx, _, _, _ = self.rx_delays.shape

        seq = const_metadata.context.sequence
        raw_seq = const_metadata.context.raw_sequence
        probe_model = get_unique_probe_model(const_metadata)

        if self.output_type == "hri":
            self._kernel = self._kernel_module.get_function("delayAndSumLutHri")
            output_shape = (self.n_seq, self.y_size, self.x_size, self.z_size)
        elif self.output_type == "lri":
            self._kernel = self._kernel_module.get_function("delayAndSumLutLri")
            output_shape = (self.n_seq, self.n_tx, self.y_size, self.x_size, self.z_size)
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")

        downsampling_factor = raw_seq.ops[0].rx.downsampling_factor
        start_sample = raw_seq.ops[0].rx.sample_range[0]

        if downsampling_factor != 1:
            raise ValueError("Currently only downsampling factor == 1 "
                             f"is supported (got: {downsampling_factor})")
        if start_sample != 0:
            raise ValueError("Currently only start sample == 0 "
                             f"is supported (got: {start_sample})")
        acq_fs = (const_metadata.context.device.sampling_frequency / downsampling_factor)
        self.output_buffer = self.num_pkg.zeros(output_shape, dtype=self.num_pkg.complex64)
        y_block_size = min(self.y_size, 8)
        x_block_size = min(self.x_size, 8)
        z_block_size = min(self.z_size, 8)
        self.block_size = (z_block_size, x_block_size, y_block_size)
        self.grid_size = (int((self.z_size - 1) // z_block_size + 1),
                          int((self.x_size - 1) // x_block_size + 1),
                          int((self.y_size - 1) // y_block_size + 1))
        self.tx_delays = self.num_pkg.asarray(self.tx_delays, dtype=self.num_pkg.float32)
        self.rx_delays = self.num_pkg.asarray(self.rx_delays, dtype=self.num_pkg.float32)
        self.tx_apodization = self.num_pkg.asarray(self.tx_apodization, dtype=self.num_pkg.uint8)
        self.rx_apodization = self.num_pkg.asarray(self.rx_apodization, dtype=self.num_pkg.float32)
        # System and transmit properties.
        pulse = raw_seq.ops[0].tx.excitation
        self.fs = self.num_pkg.float32(const_metadata.data_description.sampling_frequency)
        self.fn = self.num_pkg.float32(pulse.center_frequency)

        self.n_elements = probe_model.n_elements
        burst_factor = pulse.n_periods / (2 * self.fn)
        self.initial_delay = -start_sample / 65e6 + burst_factor
        self.initial_delay = self.num_pkg.float32(self.initial_delay)
        return const_metadata.copy(input_shape=output_shape)

    def process(self, data):
        data = self.num_pkg.ascontiguousarray(data)
        params = (
            self.output_buffer,
            data,
            self.tx_delays, self.rx_delays,
            self.tx_apodization, self.rx_apodization,
            self.initial_delay,
            self.n_tx, self.n_samples, self.n_rx,
            self.y_size, self.x_size, self.z_size,
            self.fs, self.fn)
        self._kernel(self.grid_size, self.block_size, params)
        return self.output_buffer


class RunForDlPackCapsule(Operation):
    """
    Converts input cupy ndarray into the DL Pack capsule
    (https://github.com/dmlc/dlpack) and runs the provided callback function.

    The callback function should take DL Pack capsule as input and return
    a new cupy array.

    Note: experimental.
    """

    def __init__(self, callback, name=None):
        super().__init__(name=name)
        self.callback = callback

    def prepare(self, const_metadata):
        return const_metadata

    def process(self, data):
        dlpack_capsule = data.toDlpack()
        return self.callback(dlpack_capsule)


def get_unique_probe_model(const_metadata):
    seq = const_metadata.context.sequence
    if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
        if seq.tx_placement != seq.rx_placement:
            raise ValueError("TX and RX should be done on the same Probe.")
        placement = seq.tx_placement
    elif isinstance(seq, arrus.ops.us4r.TxRxSequence):
        placements_tx = {op.tx.placement for op in seq.ops}
        placements_rx = {op.rx.placement for op in seq.ops}
        placements = placements_tx + placements_rx
        if len(placements) != 1:
            raise ValueError("TX and RX should be done on the same Probe.")
        placement = next(iter(placements))
    else:
        raise ValueError(f"Unsupported sequence type: {seq}")
    return const_metadata.context.device.get_probe_by_id(placement).model
