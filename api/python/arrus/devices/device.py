""" Arrus device handle. """
import abc
import dataclasses
import arrus.core


@dataclasses.dataclass(frozen=True)
class DeviceType:
    type: str
    core_repr: object


# Currently available python devices.
CPU = DeviceType("CPU", arrus.core.DeviceType_CPU)
Us4R = DeviceType("Us4R", arrus.core.DeviceType_Us4R)


@dataclasses.dataclass(frozen=True)
class DeviceId:
    device_type: object
    ordinal: int


class Device(abc.ABC):
    """
    A handle to device.

    This is an abstract class and should not be instantiated.
    """

    @abc.abstractmethod
    def get_device_id(self) -> DeviceId:
        pass

    def __str__(self):
        return self.get_device_id()

    def __repr__(self):
        return self.__str__()


class UltrasoundDeviceDTO(abc.ABC):
    @abc.abstractmethod
    def get_id(self):
        pass
