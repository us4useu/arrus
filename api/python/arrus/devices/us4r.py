import dataclasses
import numpy as np
import time
import ctypes
import collections.abc

import arrus.utils.core
import arrus.logging
import arrus.core
from arrus.devices.device import Device, DeviceId, DeviceType
import arrus.exceptions
import arrus.devices.probe
import arrus.metadata
import arrus.kernels
import arrus.kernels.kernel
import arrus.kernels.tgc


DEVICE_TYPE = DeviceType("Us4R", arrus.core.DeviceType_Us4R)


@dataclasses.dataclass(frozen=True)
class FrameChannelMapping:
    """
    Stores information how to get logical order of the data from
    the physical order provided by the us4r device.

    :param frames: a mapping: (logical frame, logical channel) -> physical frame
    :param channels: a mapping: (logical frame, logical channel) -> physical channel
    """
    frames: np.ndarray
    channels: np.ndarray
    batch_size: int = 1


class HostBuffer:
    """
    Buffer storing data that comes from the us4r device.

    The buffer is implemented as a circular queue. The consumer gets data from
    the queue's tails (end), the producer puts new data at the queue's head
    (firt element of the queue).

    This class provides an access to the queue's tail only. The user
    can access the latest data produced by the device by accessing `tail()`
    function. To release the tail data that is not needed anymore the user
    can call `release_tail()` function.
    """

    def __init__(self, buffer_handle,
                 fac: arrus.metadata.FrameAcquisitionContext,
                 data_description: arrus.metadata.EchoDataDescription,
                 frame_shape: tuple,
                 rx_batch_size: int):
        self.buffer_handle = buffer_handle
        self.fac = fac
        self.data_description = data_description
        self.frame_shape = frame_shape
        self.buffer_cache = {}
        self.frame_metadata_cache = {}
        # Required to determine time step between frame metadata positions.
        self.n_samples = fac.raw_sequence.get_n_samples()
        if len(self.n_samples) > 1:
            raise RuntimeError
        self.n_samples = next(iter(self.n_samples))
        # FIXME This won't work when the the rx aperture has to be splitted to multiple operations
        # Currently works for rx aperture <= 64 elements
        self.n_triggers = self.data_description.custom["frame_channel_mapping"].frames.shape[0]
        self.rx_batch_size = rx_batch_size

    def tail(self, timeout=None):
        """
        Returns data available at the tail of the buffer.

        :param timeout: timeout in milliseconds, None means infinite timeout
        :return: a pair: RF data, metadata
        """
        data_addr = self.buffer_handle.tailAddress(
            -1 if timeout is None else timeout)
        if data_addr not in self.buffer_cache:
            array = self._create_array(data_addr)
            frame_metadata_view = array[:self.n_samples*self.n_triggers*self.rx_batch_size:self.n_samples]
            self.buffer_cache[data_addr] = array
            self.frame_metadata_cache[data_addr] = frame_metadata_view
        else:
            array = self.buffer_cache[data_addr]
            frame_metadata_view = self.frame_metadata_cache[data_addr]
        # TODO extract first lines from each frame lazily
        metadata = arrus.metadata.Metadata(
            context=self.fac,
            data_desc=self.data_description,
            custom={"frame_metadata_view": frame_metadata_view}
        )
        return array, metadata

    def head(self, timeout=None):
        """
        Returns data available at the tail of the buffer.

        :param timeout: timeout in milliseconds, None means infinite timeout
        :return: a pair: RF data, metadata
        """
        data_addr = self.buffer_handle.headAddress(
            -1 if timeout is None else timeout)
        if data_addr not in self.buffer_cache:
            array = self._create_array(data_addr)
            frame_metadata_view = array[:self.n_samples*self.n_triggers*self.rx_batch_size:self.n_samples]
            self.buffer_cache[data_addr] = array
            self.frame_metadata_cache[data_addr] = frame_metadata_view
        else:
            array = self.buffer_cache[data_addr]
            frame_metadata_view = self.frame_metadata_cache[data_addr]
        # TODO extract first lines from each frame lazily
        metadata = arrus.metadata.Metadata(
            context=self.fac,
            data_desc=self.data_description,
            custom={"frame_metadata_view": frame_metadata_view}
        )
        return array, metadata

    def release_tail(self, timeout=None):
        """
        Marks the tail data as no longer needed.

        :param timeout: timeout in milliseconds, None means infinite timeout
        """
        self.buffer_handle.releaseTail(-1 if timeout is None else timeout)

    def _create_array(self, addr):
        ctypes_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int16))
        arr = np.ctypeslib.as_array(ctypes_ptr, shape=self.frame_shape)
        return arr


class Us4R(Device):
    """
    A handle to Us4R device.

    Wraps an access to arrus.core.Us4R object.
    """

    def __init__(self, handle, parent_session):
        super().__init__()
        self._handle = handle
        self._session = parent_session
        self._device_id = DeviceId(DEVICE_TYPE,
                                   self._handle.getDeviceId().getOrdinal())
        # Context for the currently running sequence.
        self._current_sequence_context = None

    def get_device_id(self):
        return self._device_id

    def set_tgc(self, tgc_curve):
        """
        Sets TGC samples for given TGC description.

        :param samples: a given TGC to set.
        """
        if isinstance(tgc_curve, arrus.ops.tgc.LinearTgc):
            if self._current_sequence_context is None:
                raise ValueError("There is no tx/rx sequence currently "
                                 "uploaded.")
            tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
                self._current_sequence_context)
        elif not isinstance(tgc_curve, collections.abc.Iterable):
            raise ValueError(f"Unrecognized tgc type: {type(tgc_curve)}")
        self._handle.setTgcCurve(list(tgc_curve))

    def set_hv_voltage(self, voltage):
        """
        Enables HV and sets a given voltage.

        :param voltage: voltage to set
        """
        self._handle.setVoltage(voltage)

    def start(self):
        """
        Starts uploaded tx/rx sequence execution.
        """
        self._handle.start()

    def stop(self):
        """
        Stops tx/rx sequence execution.
        """
        self._handle.stop()

    @property
    def sampling_frequency(self):
        """
        Device sampling frequency [Hz].
        """
        # TODO use sampling frequency from the us4r device
        return 65e6

    def upload(self, seq: arrus.ops.Operation, mode="sync",
               rx_buffer_size=None, host_buffer_size=None,
               frame_repetition_interval=None,
               rx_batch_size=1) -> HostBuffer:
        """
        Uploads a given sequence of operations to perform on the device.

        The host buffer returns Frame Channel Mapping in frame
        acquisition context custom data dictionary.

        :param seq: sequence to set
        :param mode: mode to set (str), available values: "sync", "async"
        :param rx_buffer_size: the size of the buffer to set on the Us4R device,
          should be None for "sync" version (value 2 will be used)
        :param host_buffer_size: the size of the buffer to create on the host
          computer, should be None for "sync" version (value 2 will be used)
        :param frame_repetition_interval: the expected time between successive
          frame acquisitions to set, should be None for "sync" version. None
          value means that no interval should be set
        :param rx_batch_size: number of RF frames that should be acquired in a single run
        :raises: ValueError when some of the input parameters are invalid
        :return: a data buffer
        """
        # Verify the input parameters.
        if mode not in {"async", "sync"}:
            raise ValueError(f"Unrecognized mode: {mode}")

        if mode == "sync" and (rx_buffer_size is not None
                               or frame_repetition_interval is not None):
            raise ValueError("rx_buffer_size and "
                             "frame_repetition_interval should be None "
                             "for 'sync' mode.")

        if host_buffer_size is None:
            host_buffer_size = 2

        if host_buffer_size % rx_batch_size != 0:
            raise ValueError("Host buffer size should be a multiple "
                             "of rx batch size.")
        host_buffer_size = host_buffer_size // rx_batch_size

        # Prepare sequence to load
        kernel_context = self._create_kernel_context(seq)
        self._current_sequence_context = kernel_context
        raw_seq = arrus.kernels.get_kernel(type(seq))(kernel_context)
        core_seq = arrus.utils.core.convert_to_core_sequence(raw_seq)

        # Load the sequence
        upload_result = None
        if mode == "sync":
            upload_result = self._handle.uploadSync(core_seq, host_buffer_size,
                                                    rx_batch_size)
        elif mode == "async":
            upload_result = self._handle.uploadAsync(
                core_seq, rxBufferSize=rx_buffer_size,
                hostBufferSize=host_buffer_size,
                frameRepetitionInterval=frame_repetition_interval)

        # Prepare data buffer and constant context metadata
        fcm, buffer_handle = upload_result[0], upload_result[1]

        # -- Constant metadata
        # --- FCM
        fcm_frame, fcm_channel = arrus.utils.core.convert_fcm_to_np_arrays(fcm)
        fcm = FrameChannelMapping(frames=fcm_frame, channels=fcm_channel,
                                  batch_size=rx_batch_size)

        # --- Frame acquisition context
        fac = self._create_frame_acquisition_context(seq, raw_seq)
        echo_data_description = self._create_data_description(raw_seq, fcm)

        # --- Data buffer
        n_samples = raw_seq.get_n_samples()
        if len(n_samples) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                "Currently only a sequence with constant number of samples "
                "can be accepted.")
        n_samples = next(iter(n_samples))
        return HostBuffer(
            buffer_handle=buffer_handle,
            fac=fac,
            data_description=echo_data_description,
            frame_shape=self._get_physical_frame_shape(fcm, n_samples,rx_batch_size=rx_batch_size),
            rx_batch_size=rx_batch_size)

    def _create_kernel_context(self, seq):
        return arrus.kernels.kernel.KernelExecutionContext(
            device=self._get_dto(),
            medium=self._session.get_session_context().medium,
            op=seq, custom={})

    def _create_frame_acquisition_context(self, seq, raw_seq):
        return arrus.metadata.FrameAcquisitionContext(
            device=self._get_dto(), sequence=seq, raw_sequence=raw_seq,
            medium=self._session.get_session_context().medium,
            custom_data={})

    def _create_data_description(self, raw_seq, fcm):
        return arrus.metadata.EchoDataDescription(
            sampling_frequency=self.sampling_frequency /
                               raw_seq.ops[0].rx.downsampling_factor,
            custom={"frame_channel_mapping": fcm}
        )

    def _get_physical_frame_shape(self, fcm, n_samples, n_channels=32,
                                  rx_batch_size=1):
        # TODO: We assume here, that each frame has the same number of samples!
        # This might not be case in further improvements.
        n_frames = np.max(fcm.frames) + 1
        return n_frames * n_samples * rx_batch_size, n_channels

    def _get_dto(self):
        probe_model = arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(0).getModel())
        probe_dto = arrus.devices.probe.ProbeDTO(model=probe_model)
        return Us4RDTO(probe=probe_dto, sampling_frequency=65e6)


# ------------------------------------------ LEGACY MOCK



@dataclasses.dataclass(frozen=True)
class Us4RDTO(arrus.devices.device.UltrasoundDeviceDTO):
    probe: arrus.devices.probe.ProbeDTO
    sampling_frequency: float

    def get_id(self):
        return "Us4R:0"
