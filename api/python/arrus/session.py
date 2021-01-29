import abc
import numpy as np
import importlib
import importlib.util
import dataclasses

import arrus.core
import arrus.exceptions
import arrus.devices.us4r
import arrus.medium
import arrus.metadata
import arrus.params
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.ops.us4r
import arrus.ops.imaging
import arrus.kernels.kernel
import arrus.utils
import arrus.framework


class AbstractSession(abc.ABC):
    """
    An abstract class of session.

    This class is not intended to be instantiated.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_device(self, id: str):
        """
        Returns a device located at given path.

        :param id: a path to a device, for example '/Us4R:0'
        :return: a device located in a given path.
        """
        raise ValueError("Tried to access an abstract method.")

    @abc.abstractmethod
    def set_current_medium(self, medium: arrus.medium.Medium):
        """
        Sets a medium in the current session context.

        :param medium: medium description to set
        """
        raise ValueError("Tried to access an abstract method.")


@dataclasses.dataclass(frozen=True)
class SessionContext:
    medium: arrus.medium.Medium


class Session(AbstractSession):
    """
    A communication session with the ultrasound system.
    Currently, only localhost session is available.
    """

    def __init__(self, cfg_path: str = None,
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
        self._current_processing = None

    def upload(self, scheme: arrus.ops.us4r.Scheme):
        """
        Uploads a given sequence of operations to perform on the device.

        :param scheme: scheme to upload
        :raises: ValueError when some of the input parameters are invalid
        :return: a data buffer and constant metadata
        """
        # Verify the input parameters.
        # Prepare sequence to load
        us_device = self.get_device("/Us4R:0")
        us_device_dto = us_device._get_dto()
        medium = self._context.medium
        seq = scheme.tx_rx_sequence
        processing = scheme.processing

        kernel_context = self._create_kernel_context(seq, us_device_dto, medium)
        raw_seq = arrus.kernels.get_kernel(type(seq))(kernel_context)

        actual_scheme = dataclasses.replace(scheme, tx_rx_sequence=raw_seq)
        core_scheme = arrus.utils.core.convert_to_core_scheme(actual_scheme)
        upload_result = self._session_handle.upload(core_scheme)

        # Prepare data buffer and constant context metadata
        fcm = arrus.core.getFrameChannelMapping(upload_result)
        buffer_handle = arrus.core.getFifoLockFreeBuffer(upload_result)

        ###
        # -- Constant metadata
        # --- FCM
        fcm_frame, fcm_channel = arrus.utils.core.convert_fcm_to_np_arrays(fcm)
        fcm = arrus.devices.us4r.FrameChannelMapping(
            frames=fcm_frame, channels=fcm_channel, batch_size=1)

        # --- Frame acquisition context
        fac = self._create_frame_acquisition_context(seq, raw_seq, us_device_dto, medium)
        echo_data_description = self._create_data_description(raw_seq, us_device_dto, fcm)

        # --- Data buffer
        n_samples = raw_seq.get_n_samples()

        if len(n_samples) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                "Currently only a sequence with constant number of samples "
                "can be accepted.")
        n_samples = next(iter(n_samples))
        input_shape = self._get_physical_frame_shape(fcm, n_samples, rx_batch_size=1)

        buffer = arrus.framework.DataBuffer(buffer_handle)


        const_metadata = arrus.metadata.ConstMetadata(
            context=fac, data_desc=echo_data_description,
            input_shape=input_shape, is_iq_data=False, dtype="int16")

        # numpy/cupy processing initialization
        if processing is not None:
            # setup processing
            import arrus.utils.imaging as _imaging
            if not isinstance(processing, _imaging.Pipeline):
                raise ValueError("Currently only arrus.utils.imaging.Pipeline "
                                 "processing is supported only.")
            processing.register_host_buffer(buffer)
            processing.initialize(const_metadata)

            self._current_processing = processing

            def processing_callback(element):
                print(element)
                processing(element.data)

            buffer.append_on_new_data_callback(processing_callback)

        return buffer, const_metadata

    def start_scheme(self):
        self._session_handle.startScheme()

    def stop_scheme(self):
        self._session_handle.stopScheme()
        self._current_processing.stop()

    def get_device(self, path: str):
        """
        Returns a device identified by a given id.

        The available devices are determined by the initial session settings.

        The handle to device is invalid after the session is closed
        (i.e. the session object is disposed).

        :param path: a path to the device
        :return: a handle to device
        """
        # TODO use only device available in arrus core
        # First, try getting initially available devices.
        if path in self._py_devices:
            return self._py_devices[path]
        # Then try finding a device using arrus core implementation.

        device_handle = self._session_handle.getDevice(path)

        # Cast device to its type class.
        device_id = device_handle.getDeviceId()
        device_type = device_id.getDeviceType()
        specific_device_cast = {
            # Currently only us4r is supported.
            arrus.core.DeviceType_Us4R:
                lambda handle: arrus.devices.us4r.Us4R(
                    arrus.core.castToUs4r(handle),
                    self)

        }.get(device_type, None)
        if specific_device_cast is None:
            raise arrus.exceptions.DeviceNotFoundError(path)
        specific_device = specific_device_cast(device_handle)
        # TODO(pjarosik) key should be an id, not the whole path
        self._py_devices[path] = specific_device
        return specific_device

    def get_session_context(self):
        return self._context

    def set_current_medium(self, medium: arrus.medium.Medium):
        # TODO mutex, forbid when context is frozen (e.g. when us4r is running)
        raise RuntimeError("NYI")

    def _create_py_devices(self):
        devices = {}
        # Create CPU and GPU devices
        devices["/CPU:0"] = arrus.devices.cpu.CPU(0)
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is not None:
            import cupy
            cupy.cuda.device.Device(0).use()
            devices["/GPU:0"] = arrus.devices.gpu.GPU(0)
        return devices

    def _create_kernel_context(self, seq, device, medium):
        return arrus.kernels.kernel.KernelExecutionContext(
            device=device, medium=medium, op=seq, custom={})

    def _create_frame_acquisition_context(self, seq, raw_seq, device, medium):
        return arrus.metadata.FrameAcquisitionContext(
            device=device, sequence=seq, raw_sequence=raw_seq,
            medium=medium, custom_data={})

    def _create_data_description(self, raw_seq, device, fcm):
        return arrus.metadata.EchoDataDescription(
            sampling_frequency=device.sampling_frequency /
                               raw_seq.ops[0].rx.downsampling_factor,
            custom={"frame_channel_mapping": fcm}
        )

    def _get_physical_frame_shape(self, fcm, n_samples, n_channels=32,
                                  rx_batch_size=1):
        # TODO: We assume here, that each frame has the same number of samples!
        # This might not be case in further improvements.
        n_frames = np.max(fcm.frames) + 1
        return n_frames * n_samples * rx_batch_size, n_channels



