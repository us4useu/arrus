import dataclasses
import abc

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
class EchoDataDescription(DataDescription):
    sampling_frequency: float


class Metadata:
    """
    Metadata describing the acquired data.

    This class is immutable.

    :param context: frame acquisition context
    :param data_desc: data characteristic
    :param custom: custom frame data (e.g. trigger counters, etc.)
    """
    def __init__(self, context: FrameAcquisitionContext,
                 data_desc: DataDescription, custom: dict):
        self._context = context
        self._data_char = data_desc
        # TODO make _custom_data immutable
        self._custom_data = custom

    @property
    def context(self) -> FrameAcquisitionContext:
        return self._context

    @property
    def data_description(self) -> DataDescription:
        return self._data_char

    @property
    def custom(self):
        return self._custom_data

    def copy(self, **kwargs):
        # TODO validate kwargs
        kw = dict(context=self.context, data_desc=self.data_description,
                  custom=self.custom)
        return Metadata(**{**kw, **kwargs})

