import dataclasses
import logging
import numpy as np
import time

from arrus.devices.device import Device
import arrus.devices.probe
import arrus.metadata

_logger = logging.getLogger(__name__)


class MockFileBuffer:
    def __init__(self, dataset: np.ndarray, metadata):
        self.dataset = dataset
        self.n_frames, _, _, _ = dataset.shape
        self.i = 0
        self.counter = 0
        self.metadata = metadata

    def pop(self):
        i = self.i
        self.i = (i+1) % self.n_frames
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
        self.log(logging.DEBUG, f"Loading sequence: {sequence}")
        self.buffer = MockFileBuffer(self.dataset, self.metadata)
        return self.buffer

    def start(self):
        self.log(logging.DEBUG, "Starting device.")

    def stop(self):
        self.log(logging.DEBUG, "Stopping device.")

    def set_hv_voltage(self, voltage):
        self.log(logging.DEBUG, f"Setting HV voltage {voltage}")

    def disable_hv(self):
        self.log(logging.DEBUG, f"Disabling HV.")

@dataclasses.dataclass(frozen=True)
class Us4RDTO:
    probe: arrus.devices.probe.ProbeDTO

    def get_id(self):
        return "Us4R:0"
