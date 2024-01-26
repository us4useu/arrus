import ctypes
from arrus.devices.device import Device, DeviceId, DeviceType
import arrus.core
import numpy as np

DEVICE_TYPE = DeviceType("Us4OEM")


class HVPSMeasurement:
    """
    HVPS measurement.
    """
    def __init__(self, hvps_measurement_core):
        parameters = [
            ("PLUS", 0, "VOLTAGE"),
            ("PLUS", 0, "CURRENT"),
            ("PLUS", 1, "VOLTAGE"),
            ("PLUS", 1, "CURRENT"),
            ("MINUS",0, "VOLTAGE"),
            ("MINUS",0, "CURRENT"),
            ("MINUS",1, "VOLTAGE"),
            ("MINUS",1, "CURRENT"),
        ]
        self._values = {}
        self._array = []
        for p in parameters:
            polarity, level, unit = p
            polarity = self._polarity_str2enum(polarity)
            unit = self._unit_str2enum(unit)
            m = hvps_measurement_core.get(polarity, level, unit)
            self._values[p] = m
            self._array.append(m)
        self._array = np.stack(self._array)
        self._array = self._array.reshape(2, 2, 2, -1)

    def get(self, polarity: str, level: int, unit: str):
        return self._values[(polarity.upper(), level, unit.upper())]

    def get_array(self) -> np.ndarray:
        """
        Returns the measurement as numpy array.
        The output shape is (polarity, level, unit, sample)
        """
        return self._array

    def _polarity_str2enum(self, value: str):
        return {
            "PLUS": arrus.core.HVPSMeasurement.PLUS,
            "MINUS": arrus.core.HVPSMeasurement.MINUS
        }[value]

    def _unit_str2enum(self, value: str):
        return {
            "VOLTAGE": arrus.core.HVPSMeasurement.VOLTAGE,
            "CURRENT": arrus.core.HVPSMeasurement.CURRENT
        }[value]



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

    def get_hvps_measurement(self) -> HVPSMeasurement:
        return HVPSMeasurement(self._handle.getHVPSMeasurement())

    def set_hvps_sync_measurement(self, n_samples: int, frequency: float) -> float:
        return self._handle.setHVPSSyncMeasurement(n_samples, frequency)
