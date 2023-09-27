import dataclasses
import abc

import arrus.devices.device
import arrus.medium
import arrus.ops
import arrus.ops.us4r


@dataclasses.dataclass(frozen=True)
class FrameAcquisitionContext:
    """
    Metadata describing RF frame acquisition process.

    :param device: ultrasound device specification
    :param sequence: a sequence that the user wanted to execute on the device
    :param raw_sequence: an actual Tx/Rx sequence that was uploaded on the system
    :param medium: description of the Medium assumed during communication session with the device
    :param custom_data: a dictionary with custom data
    """
    device: arrus.devices.device.UltrasoundDeviceDTO
    sequence: arrus.ops.Operation
    raw_sequence: arrus.ops.us4r.TxRxSequence
    medium: arrus.medium.MediumDTO
    custom_data: dict


class DataDescription(abc.ABC):
    pass


@dataclasses.dataclass(frozen=True)
class EchoDataDescription(DataDescription):
    """
    Data description of the ultrasound echo data.

    :param sampling_frequency: a sampling frequency of the data
    :param custom: custom information
    """
    sampling_frequency: float
    custom: dict = dataclasses.field(default_factory=dict)


class ConstMetadata:
    """

    Metadata describing the acquired data.

    This class is immutable.

    :param context: frame acquisition context
    :param data_desc: data characteristic
    :param custom: custom frame data (e.g. trigger counters, etc.)
    :param input_shape: input shape of the data, a tuple
    :param is_iq_data: true if we have bandpass data, false otherwise
    :param dtype: data type
    """
    def __init__(self, context: FrameAcquisitionContext,
                 data_desc: DataDescription,
                 input_shape, is_iq_data, dtype,
                 version=None):
        self._context = context
        self._data_char = data_desc
        self._input_shape = input_shape
        self._is_iq_data = is_iq_data
        self._dtype = dtype
        self._version = version

    @property
    def context(self) -> FrameAcquisitionContext:
        return self._context

    @property
    def data_description(self) -> DataDescription:
        return self._data_char

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def is_iq_data(self):
        return self._is_iq_data

    @property
    def dtype(self):
        return self._dtype

    @property
    def version(self):
        return self._version

    def __setstate__(self, data):
        self.__dict__ = data
        # Default version is None for packages < 0.7.0
        if "_version" not in data:
            self._version = None

    def copy(self, **kwargs):
        kw = dict(context=self.context, data_desc=self.data_description,
                input_shape=self.input_shape, is_iq_data=self.is_iq_data,
                dtype=self.dtype)
        return ConstMetadata(**{**kw, **kwargs})


class Metadata:
    """
    Metadata describing the acquired data.

    This class is immutable.

    :param context: frame acquisition context
    :param data_desc: data characteristic
    :param custom: custom frame data (e.g. trigger counters, etc.)
    """
    def __init__(self,context: FrameAcquisitionContext,
                 data_desc: DataDescription,
                 custom: dict):
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

