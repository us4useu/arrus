import abc
import queue
import copy

import numpy as np
import importlib
import importlib.util
import dataclasses

import arrus.core
import arrus.exceptions
import arrus.devices.us4r
import arrus.devices.file
import arrus.medium
import arrus.metadata
import arrus.params
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.ops.us4r
import arrus.ops.imaging
import arrus.ops.tgc
import arrus.kernels.tgc
import arrus.kernels.kernel
import arrus.utils
import arrus.utils.imaging
import arrus.utils.core
import arrus.framework
from typing import Sequence, Dict, Iterable
from numbers import Number

from arrus.devices.ultrasound import Ultrasound
from arrus.devices.us4r import Us4R


class AbstractSession(abc.ABC):
    """
    An abstract class of session.

    This class is not intended to be instantiated.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_device(self, path: str):
        """
        Returns a device located at given path.

        :param path: a path to a device, for example '/Us4R:0'
        :return: a device located in a given path.
        """
        raise ValueError("Tried to access an abstract method.")


@dataclasses.dataclass(frozen=True)
class SessionContext:
    medium: arrus.medium.Medium


class Session(AbstractSession):
    """
    A communication session with the ultrasound system.

    Currently, only localhost session is available.

    This class is a context manager. All the processing to be done on the device
    on the devices should be done withing the session context.
    """

    def __init__(self, cfg_path: str = "us4r.prototxt",
                 medium: arrus.medium.Medium = None):
        """
        Session constructor.

        :param cfg_path: a path to configuration file
        :param medium: medium description to set in context
        """
        super().__init__()
        self._session_handle = arrus.core.createSessionSharedHandle(cfg_path)
        self._context = SessionContext(medium=medium)
        self._py_devices = self._create_py_devices()
        self._current_processing: arrus.utils.imaging.Processing = None
        # Current metadata (for the full sequence)
        self.metadatas = None

    def upload(self, scheme: arrus.ops.us4r.Scheme):
        """
        Uploads a given sequence on devices.

        :param scheme: scheme to upload
        :raises: ValueError when some of the input parameters are invalid
        :return: a data buffer and constant metadata
        """
        # Verify the input parameters.
        # Prepare sequence to load
        us_device: Ultrasound = self.get_device("/Ultrasound:0")
        us_device_dto = us_device.get_dto()
        medium = self._context.medium
        sequences = scheme.tx_rx_sequence
        if not isinstance(sequences, Iterable):
            sequences = (sequences, )
        processing = scheme.processing
        constants = scheme.constants

        if len(constants) > 0 and len(sequences) > 1:
            raise ValueError(
                "Currently session constants can only be provided for a "
                "single-sequence schemes."
            )

        raw_seqs = []
        tx_delay_constants = ()
        # TODO make sure all sequences have the same TGC (different TGCs are not supported)
        # Convert to raw sequences and upload.
        sequences = [dataclasses.replace(s, name=f"TxRxSequence:{i}")
                     if s.name is None else s
                     for i, s in enumerate(sequences)]
        for i, sequence in enumerate(sequences):
            kernel_context = self._create_kernel_context(
                sequence,
                us_device_dto,
                medium,
                scheme.digital_down_conversion,
                constants
            )
            conversion_results = arrus.kernels.get_kernel(type(sequence))(kernel_context)
            raw_seq = conversion_results.sequence
            raw_seqs.append(raw_seq)
            tx_delay_constants = conversion_results.constants

        actual_scheme = dataclasses.replace(
            scheme,
            tx_rx_sequence=raw_seqs,
            constants=tx_delay_constants
        )
        core_scheme = arrus.utils.core.convert_to_core_scheme(actual_scheme)
        upload_result = self._session_handle.upload(core_scheme)
        # Update the DTO with the new data sampling frequency (determined by the scheme).
        us_device_dto = dataclasses.replace(
            us_device_dto,
            data_sampling_frequency=us_device.current_sampling_frequency
        )

        # Output buffer
        buffer_handle = arrus.core.getFifoLockFreeBuffer(upload_result)
        self.buffer = arrus.framework.DataBuffer(buffer_handle)

        # Constant metadata
        # NOTE: the below should be called after session_handle.upload()
        us_device.set_tgc_and_context(sequences, self.medium)
        self.metadatas = []

        for i, (raw_seq, seq) in enumerate(zip(raw_seqs, sequences)):
            data_description = us_device.get_data_description(upload_result, raw_seq, array_id=i)
            # -- Constant metadata
            # --- Frame acquisition context
            fac = self._create_frame_acquisition_context(
                seq, raw_seq, us_device_dto, medium, tx_delay_constants)
            input_shape = self.buffer.elements[0].arrays[i].shape
            is_iq_data = scheme.digital_down_conversion is not None
            const_metadata = arrus.metadata.ConstMetadata(
                context=fac, data_desc=data_description,
                input_shape=input_shape, is_iq_data=is_iq_data, dtype="int16",
                version=arrus.__version__
            )
            self.metadatas.append(const_metadata)

        # numpy/cupy processing initialization
        return  self._set_processing(self.buffer, self.metadatas, processing, sequences)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_scheme()
        self.close()

    def start_scheme(self):
        """
        Starts the execution of the uploaded scheme.
        """
        arrus.core.arrusSessionStartScheme(self._session_handle)

    def stop_scheme(self):
        """
        Stops execution of the scheme.
        """
        arrus.core.arrusSessionStopScheme(self._session_handle)
        if self._current_processing is not None:
            self._current_processing.close()
            self._current_processing = None

    def run(self, sync: bool=False, timeout: int=None):
        """
        Runs the uploaded scheme.

        The behaviour of this method depends on the work mode:
        - MANUAL: triggers execution of batch of sequences only ONCE,
        - MANUAL_OP: triggers execution of a single TX/RX only ONCE,
        - HOST, ASYNC: triggers execution of batch of sequences IN A LOOP (Host: trigger is on buffer element release).
          The run function can be called only once (before the scheme is stopped).

        :param sync: whether this method should work in a synchronous or asynchronous; true means synchronous, i.e.
                     the caller will wait until the triggered TX/RX or sequence of TX/RXs has been done. This parameter only
                     matters when the work mode is set to MANUAL or MANUAL_OP.
        :param timeout: timeout [ms]; std::nullopt means to wait infinitely. This parameter is only relevant when
                        sync = true; the value of this parameter only matters when work mode is set to MANUAL or MANUAL_OP.
        """
        arrus.core.arrusSessionRun(self._session_handle, sync, timeout)

    def close(self):
        """
        Closes session.

        This method disconnects with all the devices available during this session.
        Sets the state of the session to closed, any subsequent call to the object
        methods (e.g. upload, startScheme..) will result in exception.
        """
        self.stop_scheme()
        if self._current_processing is not None:
            self._current_processing.close()
        self._session_handle.close()

    def get_device(self, path: str):
        """
        Returns a device identified by a given id.

        The available devices are determined by the initial session settings.

        The handle to device is invalid after the session is closed
        (i.e. the session object is disposed).

        :param path: a path to the device
        :return: a handle to device
        """
        device_handle = self._session_handle.getDevice(path)

        device_id = device_handle.getDeviceId()
        device_type = device_id.getDeviceType()
        device_ordinal = device_id.getOrdinal()

        py_id = (device_type, device_ordinal)
        if py_id in self._py_devices:
            return self._py_devices[py_id]


        # Cast device to its type class.
        specific_device_cast = {
            arrus.core.DeviceType_Us4R:
                lambda handle: arrus.devices.us4r.Us4R(
                    arrus.core.castToUs4r(handle)),
            arrus.core.DeviceType_File:
                lambda handle: arrus.devices.file.File(
                    arrus.core.castToFile(handle))
        }.get(device_type, None)
        if specific_device_cast is None:
            raise arrus.exceptions.DeviceNotFoundError(path)
        specific_device = specific_device_cast(device_handle)
        self._py_devices[py_id] = specific_device
        return specific_device

    def set_parameters(self, params):
        if self._contains_py_params(params):
            self._handle_py_params(params)
            params = self._remove_py_params(params)
        core_params = arrus.utils.core.convert_to_core_parameters(params)
        self._session_handle.setParameters(core_params)

    def set_parameter(self, key: str, value: Sequence[Number]):
        """
        Sets the value for parameter with the given name.
        TODO: note: this method currently is not thread-safe
        """
        if self._current_processing is not None:
            return self._current_processing.set_parameter(key, value)

    def get_parameter(self, key: str) -> Sequence[Number]:
        """
        Returns the current value for parameter with the given name.
        """
        if self._current_processing is not None:
            return self._current_processing.processing.get_parameter(key)

    def get_parameters(self) -> Dict[str, arrus.params.ParameterDef]:
        if self._current_processing is not None:
            return self._current_processing.get_parameters()

    def get_session_context(self):
        return self._context

    @property
    def medium(self):
        """
        Returns currently set Medium.
        NOTE: this method is not thread-safe!
        """
        return self._context.medium

    @medium.setter
    def medium(self, value):
        """
        Sets a new medium in the current session context.
        NOTE: this method is not thread-safe!
        """
        self._context = SessionContext(medium=value)

    def set_subsequence(self, start, end, array_id=0, processing=None, sri=None):
        """
        Sets the current TX/RX sequence to the [start, end] subsequence (both inclusive).

        This method requires that:

        - start <= end (when start= == end, the system will run a single TX/RX sequence),
        - the scheme was uploaded,
        - the TX/RX sequence length is greater than the `end` value,
        - the scheme is stopped.

        You can specify the new SRI with the sri parameter, if None, the total PRI will be used.

        :return: the new data buffer and metadata
        """
        metadata = self.metadatas[array_id]
        upload_result = self._session_handle.setSubsequence(start, end, sri, array_id)
        # Get the new buffer
        buffer_handle = arrus.core.getFifoLockFreeBuffer(upload_result)
        self.buffer = arrus.framework.DataBuffer(buffer_handle)
        # Create new metadata
        metadata = copy.deepcopy(metadata)
        us_device: Ultrasound = self.get_device("/Ultrasound:0")
        input_shape = self.buffer.elements[0].data.shape
        sequence = metadata.context.sequence.get_subsequence(start, end)
        raw_sequence = metadata.context.raw_sequence.get_subsequence(start, end)
        data_description = us_device.get_data_description_updated_for_subsequence(array_id, upload_result, sequence)
        fac = dataclasses.replace(
            metadata.context,
            sequence=sequence,
            raw_sequence=raw_sequence
        )
        metadata = metadata.copy(
            input_shape=input_shape,
            data_desc=data_description,
            context=fac,
        )
        return self._set_processing(self.buffer, [metadata], processing, [sequence])

    def _set_processing(self, buffer, metadatas, processing, sequences):
        # setup processing
        if self._current_processing is not None:
            self._current_processing.close()
            self._current_processing = None

        if processing is not None:
            # setup processing
            import arrus.utils.imaging as _imaging
            if not isinstance(processing, _imaging.Processing):
                # Wrap into the Processing object.
                processing = _imaging.Processing(
                    graph=processing,
                    callback=None,
                )
            processing_runner = arrus.utils.imaging.ProcessingRunner(
                input_buffer=buffer, metadata=metadatas, processing=processing,
            )
            outputs = processing_runner.outputs
            self._current_processing = processing_runner
        else:
            # Device buffer and const_metadata
            outputs = buffer, metadatas
        return outputs

    def _contains_py_params(self, params):
        # Currently only start/stop params must by handled
        # by the Python layer, because os the self._buffer handle
        return Us4R.SEQUENCE_START_VAR in params or Us4R.SEQUENCE_END_VAR in params

    def _remove_py_params(self, params):
        params = params.copy()
        params.pop(Us4R.SEQUENCE_START_VAR, None)
        params.pop(Us4R.SEQUENCE_END_VAR, None)
        return params

    def _handle_py_params(self, params):
        # Currently only start/stop params must be handled in the Python layer.
        sequence_start = params.get(Us4R.SEQUENCE_START_VAR, None)
        sequence_end = params.get(Us4R.SEQUENCE_START_VAR, None)
        self.set_subsequence(sequence_start, sequence_end)

    # def set_current_medium(self, medium: arrus.medium.Medium):
    #     # TODO mutex, forbid when context is frozen (e.g. when us4r is running)
    #     raise RuntimeError("NYI")

    def _create_py_devices(self):
        devices = {
            (arrus.core.DeviceType_CPU, 0) : arrus.devices.cpu.CPU(0)
        }
        # Create CPU and GPU devices
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is not None:
            import cupy
            cupy.cuda.device.Device(0).use()
            devices[(arrus.core.DeviceType_GPU, 0)] = arrus.devices.gpu.GPU(0)
        return devices

    def _create_kernel_context(self, seq, device, medium, hardware_ddc,
                               constants):
        return arrus.kernels.kernel.KernelExecutionContext(
            device=device, medium=medium, op=seq, custom={},
            hardware_ddc=hardware_ddc,
            constants=constants
        )

    def _create_frame_acquisition_context(self, seq, raw_seq, device, medium,
                                          constants):
        return arrus.metadata.FrameAcquisitionContext(
            device=device, sequence=seq, raw_sequence=raw_seq,
            medium=medium, custom_data={},
            constants=constants
        )
