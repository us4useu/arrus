import dataclasses
import numpy as np
import time
import ctypes

import arrus.utils.core
import arrus.logging
import arrus.core
from arrus.devices.device import Device, DeviceId, DeviceType
import arrus.exceptions
import arrus.devices.probe
import arrus.metadata
import arrus.kernels
import arrus.kernels.kernel


DEVICE_TYPE = DeviceType("Us4R", arrus.core.DeviceType_Us4R)


@dataclasses.dataclass(frozen=True)
class FrameChannelMapping:
    """
    Stores information how to get logical order of the data from
    the physical order provided by the us4r device.

    :param frames: a mapping: (logical frame, logical channel)->physical frame
    :param channels: a mapping: (logical frame, logical channel)->physical channel
    """
    frames: np.ndarray
    channels: np.ndarray


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
                 frame_shape: tuple):
        self.buffer_handle = buffer_handle
        self.fac = fac
        self.data_description = data_description
        self.frame_shape = frame_shape
        self.buffer_cache = {}

    def tail(self):
        """
        Returns data available at the tail of the buffer.

        :return: a pair: RF data, metadata
        """
        data_addr = self.buffer_handle.tailAddress()
        if data_addr not in self.buffer_cache:
            array = self._create_array(data_addr)
            self.buffer_cache[data_addr] = array
        else:
            array = self.buffer_cache[data_addr]
        # TODO extract first lines from each frame lazily
        metadata = arrus.metadata.Metadata(
            context=self.fac,
            data_desc=self.data_description,
            custom={}
        )
        return array, metadata

    def release_tail(self):
        """
        Marks the tail data as no longer needed.
        """
        self.buffer_handle.releaseTail()

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

    def get_device_id(self):
        return self._device_id

    def set_voltage(self, voltage):
        """
        Enables HV and sets a given voltage.

        :param voltage: voltage to set
        """
        self._handle.setVoltage(voltage)

    def disable_hv(self):
        """
        Disables high voltage supplier.
        """
        self._handle.disableHV()

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

    def upload(self, seq: arrus.ops.Operation) -> HostBuffer:
        """
        Uploads a given sequence of operations to perform on the device.

        The host buffer returns Frame Channel Mapping in frame
        acquisition context custom data dictionary.

        :param seq: sequence to set
        :return: a data buffer
        """
        # Prepare sequence to load
        kernel_context = self._create_kernel_context(seq)
        raw_seq = arrus.kernels.get_kernel(type(seq))(kernel_context)
        core_seq = arrus.utils.core.convert_to_core_sequence(raw_seq)

        # Load the sequence
        upload_result = self._handle.uploadSync(core_seq)

        # Prepare data buffer and constant context metadata
        fcm, buffer_handle = upload_result[0], upload_result[1]

        # -- Constant metadata
        # --- FCM
        fcm_frame, fcm_channel = arrus.utils.core.convert_fcm_to_np_arrays(fcm)
        fcm = FrameChannelMapping(frames=fcm_frame, channels=fcm_channel)

        # --- Frame acquisition context
        fac = self._create_frame_acquisition_context(seq, raw_seq)
        echo_data_description = self._create_data_description(raw_seq, fcm)

        # --- Data buffer
        n_samples = raw_seq.get_n_samples()
        if len(n_samples) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                "Currently only a sequence with contant number of samples "
                "can be accepted.")
        n_samples = next(iter(n_samples))
        return HostBuffer(
            buffer_handle=buffer_handle,
            fac=fac,
            data_description=echo_data_description,
            frame_shape=self._get_physical_frame_shape(fcm, n_samples))

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

    def _get_physical_frame_shape(self, fcm, n_samples, n_channels=32):
        # TODO: We assume here, that each frame has the same number of samples!
        # This might not be case in further improvements.
        n_frames = np.max(fcm.frames) + 1
        return n_frames * n_samples, n_channels

    def _get_dto(self):
        probe_model = arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(0).getModel())
        probe_dto = arrus.devices.probe.ProbeDTO(model=probe_model)
        return Us4RDTO(probe=probe_dto, sampling_frequency=65e6)


# ------------------------------------------ LEGACY MOCK
class MockFileBuffer:
    def __init__(self, dataset: np.ndarray, metadata):
        self.dataset = dataset
        self.n_frames, _, _, _ = dataset.shape
        self.i = 0
        self.counter = 0
        self.metadata = metadata

    def pop(self):
        i = self.i
        self.i = (i + 1) % self.n_frames
        custom_data = {
            "pulse_counter": self.counter,
            "trigger_counter": self.counter,
            "timestamp": time.time_ns() // 1000000
        }
        self.counter += 1

        metadata = arrus.metadata.Metadata(
            context=self.metadata.context,
            data_desc=self.metadata.data_description,
            custom=custom_data)
        return np.array(self.dataset[self.i, :, :, :]), metadata


class MockUs4R(Device):
    def __init__(self, dataset: np.ndarray, metadata, index: int):
        super().__init__("Us4R", index)
        self.dataset = dataset
        self.metadata = metadata
        self.buffer = None

    def upload(self, sequence):
        self.buffer = MockFileBuffer(self.dataset, self.metadata)
        return self.buffer

    def start(self):
        pass

    def stop(self):
        pass

    def set_hv_voltage(self, voltage):
        pass

    def disable_hv(self):
        pass


@dataclasses.dataclass(frozen=True)
class Us4RDTO(arrus.devices.device.UltrasoundDeviceDTO):
    probe: arrus.devices.probe.ProbeDTO
    sampling_frequency: float

    def get_id(self):
        return "Us4R:0"
