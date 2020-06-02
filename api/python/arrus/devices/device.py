""" ARRUS Devices. """
import logging
import abc

_logger = logging.getLogger(__name__)


class DeviceCfg(abc.ABC):
    pass


class Device:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    @staticmethod
    def get_device_id(name, index):
        if index is None:
            return name
        else:
            return "%s:%d" % (name, index)

    def get_id(self):
        return Device.get_device_id(self.name, self.index)

    def log(self, level, msg):
        _logger.log(level, "%s: %s" % (self.get_id(), msg))

    def __str__(self):
        return self.get_id()

    def __repr__(self):
        return self.__str__()
