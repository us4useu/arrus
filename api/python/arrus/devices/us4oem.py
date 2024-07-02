import ctypes
from arrus.devices.device import Device, DeviceId, DeviceType
import arrus.core

DEVICE_TYPE = DeviceType("Us4OEM")


class Us4OEM(Device):

    def __init__(self, handle):
        self._handle = handle
        self._device_id = DeviceId(DEVICE_TYPE,
                                   self._handle.getDeviceId().getOrdinal())

    def get_device_id(self) -> DeviceId:
        return self._device_id

    def get_firmware_version(self) -> int:
        """
        Returns us4OEM's main firmware version.
        """
        return ctypes.c_ulong(self._handle.getFirmwareVersion()).value

    def get_tx_firmware_version(self) -> int:
        """
        Returns TX Firmware version.
        """
        return ctypes.c_ulong(self._handle.getTxFirmwareVersion()).value

    def get_fpga_temperature(self) -> float:
        """
        Returns Us4OEM FPGA temperature [Celsius]
        """
        return arrus.core.arrusUs4OEMGetFPGATemperature(self._handle)

    def get_ucd_temperature(self) -> float:
        """
        Returns Us4OEM UCD temperature [Celsius]
        """
        return arrus.core.arrusUs4OEMGetUCDTemperature(self._handle)

    def get_ucd_external_temperature(self) -> float:
        """
        Returns Us4OEM UCD external temperature [Celsius]
        """
        return arrus.core.arrusUs4OEMGetUCDExternalTemperature(self._handle)

    def get_serial_number(self) -> str:
        """
        Returns serial number of the device.
        """
        return self._handle.getSerialNumber()

    def get_revision(self) -> str:
        """
        Returns revision number of the device.
        """
        return self._handle.getRevision()
