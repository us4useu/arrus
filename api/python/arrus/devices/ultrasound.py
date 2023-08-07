import arrus.devices.probe
from arrus.devices.device import Device, DeviceId, DeviceType
from abc import ABC, abstractmethod


DEVICE_TYPE = DeviceType("Ultrasound")


class Ultrasound(ABC):

    @abstractmethod
    def set_kernel_context(self):
        """
        This method is intended to provide any session specific data to
        the device, on the stage of sequence upload.
        """
        pass

    @abstractmethod
    def get_dto(self):
        """
        Returns descriptor of this device.
        """
        pass

    @abstractmethod
    def get_data_description(self):
        """
        DEPRECATED: This method will be removed after moving most of the
        session-related code to the ARRUS core.
        """
        pass







