import arrus.devices.device
import arrus.medium
import dataclasses
import abc


@dataclasses.dataclass(frozen=True)
class FrameAcquisitionContext:
    device: arrus.devices.device.UltrasoundDeviceDTO
    sequence: arrus.ops.Operation
    medium: arrus.medium.MediumDTO
    custom_data: dict


class DataCharacteristic(abc.ABC):
    pass


@dataclasses.dataclass(frozen=True)
class EchoSignalDataCharacteristic(DataCharacteristic):
    sampling_frequency: float


class Metadata:
    def __init__(self, context_descriptor: FrameAcquisitionContext,
                 data_char: DataCharacteristic, custom_data: dict):
        self._context = context_descriptor
        self._data_char = data_char
        # TODO make _custom_data immutable
        self._custom_data = custom_data

    def get_context(self):
        return self._context

    def get_data_characteristic(self):
        return self.data_char

    def get_custom_data(self):
        return self._custom_data