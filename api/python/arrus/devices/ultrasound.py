import arrus.devices.probe
from arrus.devices.device import Device, DeviceId, DeviceType
from abc import ABC, abstractmethod
import dataclasses


DEVICE_TYPE = DeviceType("Ultrasound")


class Ultrasound(ABC):
    """
    An interface to the ultrasound device.
    Please note that this class is pure abstract class and you
    should use the concrete implementation of it (File, Us4R, etc.).
    """

    @abstractmethod
    def set_tgc_and_context(self, sequences, medium):
        """
        This method is intended to provide any session specific data to
        the device, on the stage of sequence upload.

        # TODO(0.11.0) Deprecated: we should avoid storing linear tgc variables
        ( speed of sound, number of samples) and instead keep the information
        in the low-level software (ARRUS C++), simply the current TxRxSequences.
        Setting TGC should also have a parameter "tgc_profile" which should
        be variable that can be modified in real-time.
        """
        pass

    @abstractmethod
    def get_dto(self):
        """
        Returns descriptor of this device.
        """
        pass

    @abstractmethod
    def get_data_description(self, upload_result, sequence, array_id):
        """
        DEPRECATED: This method will be removed after moving most of the
        session-related code to the ARRUS core.
        """
        pass

    @abstractmethod
    def sampling_frequency(self):
        pass

    @abstractmethod
    def current_sampling_frequency(self):
        pass


@dataclasses.dataclass(frozen=True)
class UltrasoundDTO:
    probe: arrus.devices.probe.ProbeDTO
    sampling_frequency: float

    def get_id(self):
        return "Ultrasound:0"








