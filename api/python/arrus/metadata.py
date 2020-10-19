import dataclasses
import abc

import arrus.ops
import arrus.devices.device
import arrus.medium



@dataclasses.dataclass(frozen=True)
class FrameAcquisitionContext:
    device: arrus.devices.device.UltrasoundDeviceDTO
    sequence: arrus.ops.Operation
    medium: arrus.medium.MediumDTO
    custom_data: dict


class DataDescription(abc.ABC):
    pass


@dataclasses.dataclass(frozen=True)
class EchoSignalDataDescription(DataDescription):
    sampling_frequency: float


class Metadata:
    """
    Metadata describing the acquired data.

    This class is immutable.
    """
    def __init__(self, context: FrameAcquisitionContext,
                 data_desc: DataDescription, custom: dict):
        self._context = context
        self._data_char = data_desc
        # TODO make _custom_data immutable
        self._custom_data = custom

    @property
    def context(self):
        return self._context

    @property
    def data_description(self):
        return self._data_char

    @property
    def custom(self):
        return self._custom_data