import time
import arrus.metadata
import arrus.logging
from arrus.logging import (DEBUG)
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
        frame_metadata = np.zeros((175, 32), dtype=np.int16)
        custom_data = {
            "frame_metadata_view": frame_metadata
        }
        metadata = arrus.metadata.Metadata(
            context=self.metadata.context,
            data_desc=self.metadata.data_description,
            custom=custom_data)
        return np.array(self.dataset[self.i]), metadata

    def release_tail(self, timeout=None):
        self.i = (self.i + 1) % self.n_frames


class MockUs4R(Device):
    def __init__(self, dataset: np.ndarray, metadata, index: int):
        super().__init__()
        self.dataset = dataset
        self.metadata = metadata
        self.const_metadata = arrus.metadata.ConstMetadata(
            context=metadata.context,
            data_desc=metadata.data_description,
            input_shape=dataset[0].shape,
            is_iq_data=False,
            dtype='int16')
        self.buffer = None

    def get_device_id(self):
        return arrus.devices.device.DeviceId("Us4R", 0)

    def set_hv_voltage(self, voltage):
        """
        Enables high voltage supplier and sets a given voltage value.

        The voltage is determined by the probe specification;
        Us4R can maximally accept 90 Vpp.

        :param voltage: voltage to set [0.5*Vpp]
        """
        arrus.logging.log(DEBUG, f"Set voltage {voltage}")

    def disable_hv(self):
        """
        Disables high voltage supplier.
        """
        arrus.logging.log(DEBUG, "Disable HV voltage.")

    def start(self):
        """
        Starts uploaded tx/rx sequence execution.
        """
        arrus.logging.log(DEBUG, "Started device.")

    def stop(self):
        """
        Stops tx/rx sequence execution.
        """
        arrus.logging.log(DEBUG, "Stopped device.")

    @property
    def sampling_frequency(self):
        """
        Device sampling frequency [Hz].
        """
        # TODO use sampling frequency from the us4r device
        return 65e6

    def upload(self, seq: arrus.ops.Operation,
               rx_buffer_size=None, host_buffer_size=None,
               rx_batch_size=None):
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
        arrus.logging.log(arrus.logging.DEBUG, f"Uploaded sequence: {seq}")
        return MockFileBuffer(self.dataset, self.metadata), self.const_metadata

