import time
import arrus.metadata
import arrus
import numpy as np
from arrus.devices.device import Device


class MockFileBuffer:

    def __init__(self, dataset: np.ndarray, metadata):
        self.dataset = dataset
        self.n_frames, _, _, _ = dataset.shape
        self.i = 0
        self.counter = 0
        self.metadata = metadata


    def tail(self, timeout=None):
        custom_data = {
            "pulse_counter": self.counter,
            "trigger_counter": self.counter,
            "timestamp": time.time_ns() // 1000000
        }
        metadata = arrus.metadata.Metadata(
            context=self.metadata.context,
            data_desc=self.metadata.data_description,
            custom=custom_data)
        return np.array(self.dataset[self.i, :, :, :]), metadata


    def release_tail(self, timeout=None):
        i = self.i
        self.i = (i + 1) % self.n_frames

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

    def get_device_id(self):
        return Device

    def set_voltage(self, voltage):

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

    def upload(self, seq: arrus.ops.Operation, mode="sync",
               rx_buffer_size=None, host_buffer_size=None,
               frame_repetition_interval=None) -> MockFileBuffer:
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
        :raises: ValueError when some of the input parameters are invalid
        :return: a data buffer
        """
        # Verify the input parameters.
        if mode not in {"async", "sync"}:
            raise ValueError(f"Unrecognized mode: {mode}")

        if mode == "sync" and (rx_buffer_size is not None
                               or host_buffer_size is not None
                               or frame_repetition_interval is not None):
            raise ValueError("rx_buffer_size, host_buffer_size and "
                             "frame_repetition_interval should be None "
                             "for 'sync' mode.")

        # Prepare sequence to load
        kernel_context = self._create_kernel_context(seq)
        raw_seq = arrus.kernels.get_kernel(type(seq))(kernel_context)
        core_seq = arrus.utils.core.convert_to_core_sequence(raw_seq)

        # Load the sequence
        upload_result = None
        if mode == "sync":
            upload_result = self._handle.uploadSync(core_seq)
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