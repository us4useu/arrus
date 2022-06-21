import ctypes
from arrus.devices.device import Device, DeviceId, DeviceType

DEVICE_TYPE = DeviceType("Us4OEM")


class Us4OEM(Device):

    def __init__(self, handle):
        self._handle = handle
        self._device_id = DeviceId(DEVICE_TYPE,
                                   self._handle.getDeviceId().getOrdinal())

    def get_device_id(self) -> DeviceId:
        return self._device_id

    def get_firmware_version(self) -> int:
        return ctypes.c_ulong(self._handle.getFirmwareVersion()).value

    def get_tx_firmware_version(self) -> int:
        return ctypes.c_ulong(self._handle.getTxFirmwareVersion()).value