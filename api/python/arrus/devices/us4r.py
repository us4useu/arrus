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
        self.pulseCounter = 0
        self.metadata = metadata

    def pop(self):
        i = self.i
        self.i = (i+1) % self.n_frames
        custom_data = {
            "pulse_counter": self.pulseCounter,
            "timestamp": time.time_ns() // 1000000
        }
        self.pulseCounter += 1
        metadata = arrus.metadata.Metadata(
            context_descriptor=self.metadata.get_context(),
            data_char=self.metadata.get_data_characteristic(),
            custom_data=custom_data
        )
        return self.dataset[self.i], metadata


class MockUs4R(Device):
    def __init__(self, dataset: np.ndarray, metadata, name: str, index: int):
        super().__init__(name, index)
        self.dataset = dataset
        self.metadata = metadata
        self.buffer = None

    def upload(self, sequence):
        self.log(logging.INFO, f"Uploading sequence: {sequence}")

    def start(self):
        self.log(logging.INFO, "Starting device.")
        self.buffer = MockFileBuffer(self.dataset, self.metadata)
        return self.buffer

    def stop(self):
        self.log(logging.INFO, "Stopping device.")

    def set_hv_voltage(self, voltage):
        self.log(logging.INFO, f"Setting HV voltage {voltage}")

    def disable_hv(self):
        self.log(logging.INFO, f"Disabling HV.")

@dataclasses.dataclass(frozen=True)
class Us4RDTO:
    probe: arrus.devices.probe.ProbeDTO

    def get_id(self):
        return "Us4R:0"
