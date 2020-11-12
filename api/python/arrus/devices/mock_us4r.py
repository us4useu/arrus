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
        custom_data = {
            "pulse_counter": self.counter,
            "trigger_counter": self.counter,
            "timestamp": time.time_ns() // 1000000
        }
        metadata = arrus.metadata.Metadata(
            context=self.metadata.context,
            data_desc=self.metadata.data_description,
            custom=custom_data)
        return np.array(self.dataset[self.i]), metadata

    def release_tail(self, timeout=None):
        self.i = (self.i + 1) % self.n_frames

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

        arrus.logging.log(f"Uploaded sequence: {seq}")

