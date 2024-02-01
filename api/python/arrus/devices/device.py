""" Arrus device handle. """
import abc
import dataclasses
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class DeviceType:
    type: str


@dataclasses.dataclass(frozen=True)
class DeviceId:
    device_type: object
    ordinal: int


def split_device_id_str(s):
    """
    Extracts device type from given string id.

    :param s: id as a string
    :return: device type string
    """
    s = s.strip()
    n = s.split("/")
    if s.startswith("/"):
        n = n[1:]
    if len(n) == 0:
        raise ValueError("No device provided in the id.")
    c = n[0].split(":")
    return c[0], int(c[1])


def parse_device_id(device_id_str: str) -> DeviceId:
    id = device_id_str.strip()
    type, ordinal = id.split(":")
    ordinal = int(ordinal.strip())
    type = type.strip()
    return DeviceId(
        device_type=DeviceType(type),
        ordinal=ordinal
    )


def split_to_device_ids(path: str) -> Sequence[DeviceId]:
    path = path.strip()
    ids = path.split("/")
    if path.startswith("/"):
        ids = ids[1:]
    if len(ids) == 0:
        raise ValueError("No device provided in the id.")
    result = []
    for id in ids:
        id = id.split()
        type, ordinal = id.split(":")
        ordinal = int(ordinal)
        result.append(DeviceId(
            device_type=DeviceType(type),
            ordinal=ordinal
        ))
    return result


class Device(abc.ABC):
    """
    A handle to device.

    This is an abstract class and should not be instantiated.
    """

    @abc.abstractmethod
    def get_device_id(self) -> DeviceId:
        pass

    def __str__(self):
        return str(self.get_device_id())

    def __repr__(self):
        return self.__str__()
