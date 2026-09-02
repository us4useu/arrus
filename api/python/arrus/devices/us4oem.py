import ctypes
from arrus.devices.device import Device, DeviceId, DeviceType
import arrus.core
import numpy as np
from enum import Enum, IntEnum


DEVICE_TYPE = DeviceType("Us4OEM")


class Us4OEMInterrupt(IntEnum):
    """
    The set of us4OEM system interrupts a user callback can be registered for
    via :func:`arrus.create_session_settings_from`.
    """
    PROBE_NOT_CONNECTED = arrus.core.Us4OEMInterrupt_PROBE_NOT_CONNECTED
    PULSER_INTERRUPT = arrus.core.Us4OEMInterrupt_PULSER_INTERRUPT
    TX_TIMEOUT = arrus.core.Us4OEMInterrupt_TX_TIMEOUT
    WATCHDOG_IRQ0 = arrus.core.Us4OEMInterrupt_WATCHDOG_IRQ0
    WATCHDOG_IRQ1 = arrus.core.Us4OEMInterrupt_WATCHDOG_IRQ1
    HVPS_FUSE = arrus.core.Us4OEMInterrupt_HVPS_FUSE


#: Interrupts that, when raised, indicate the system has entered (or should
#: enter) a safe state. Includes every supported us4OEM interrupt except
#: WATCHDOG_IRQ0, which is only an early warning.
SAFE_STATE_INTERRUPTS = frozenset({
    Us4OEMInterrupt.PROBE_NOT_CONNECTED,
    Us4OEMInterrupt.PULSER_INTERRUPT,
    Us4OEMInterrupt.TX_TIMEOUT,
    Us4OEMInterrupt.WATCHDOG_IRQ1,
    Us4OEMInterrupt.HVPS_FUSE,
})


class HVPSMeasurement:
    """
    HVPS measurement.
    """
    def __init__(self, hvps_measurement_core):
        parameters = [
            ("MINUS",0, "VOLTAGE"),
            ("MINUS",0, "CURRENT"),
            ("MINUS",1, "VOLTAGE"),
            ("MINUS",1, "CURRENT"),
            ("PLUS", 0, "VOLTAGE"),
            ("PLUS", 0, "CURRENT"),
            ("PLUS", 1, "VOLTAGE"),
            ("PLUS", 1, "CURRENT"),
        ]
        self._values = {}
        self._array = []
        for p in parameters:
            polarity, level, unit = p
            polarity = self._polarity_str2enum(polarity)
            unit = self._unit_str2enum(unit)
            m = list(hvps_measurement_core.get(level, polarity, unit))
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

        polarity: 0: MINUS, 1: PLUS
        level: 0 or 1
        unit: 0: voltage, 1: current
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


class Variant(Enum):
    """
    Us4OEM variant.
    """
    LEGACY = "LEGACY"
    PLUS_RX_32 = "PLUS_RX_32"
    PLUS_RX_64 = "PLUS_RX_64"
    PLUS_HF = "PLUS_HF"


def _variant_enum_to_enum(enum):
    """
    Variant C++ Enum to Python Enum.
    """
    return {
        arrus.core.Us4OEM.Variant_LEGACY: Variant.LEGACY,
        arrus.core.Us4OEM.Variant_PLUS_RX_32: Variant.PLUS_RX_32,
        arrus.core.Us4OEM.Variant_PLUS_RX_64: Variant.PLUS_RX_64,
        arrus.core.Us4OEM.Variant_PLUS_HF: Variant.PLUS_HF,
    }[enum]


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

    def set_wait_for_hvps_measurement_done(self):
        """
        Configures the system to sync with the HVPS Measurement done irq.

        This method is intended to be used in the probe_check implementation.
        """
        return self._handle.setWaitForHVPSMeasurementDone()

    def wait_for_hvps_measurement_done(self, timeout: int=None):
        """
        Waits for the HVPS Measurement done irq.

        This method is intended to be used in the probe_check implementation.
        """
        return arrus.core.arrusUs4OEMWaitForHVPSMeasuerementDone(self._handle, timeout)

    def get_variant(self) -> Variant:
        """
        Returns variant of the device.
        """
        core_variant = self._handle.getVariant()
        return _variant_enum_to_enum(core_variant)

    def get_fpga_wallclock(self) -> int:
        """
        Returns the current FPGA wallclock value, expressed in number of
        FPGA clock periods. Divide by ``Us4R.sampling_frequency`` (or, in the
        vandv wrapper, ``device.get_sampling_frequency()``) to convert to
        seconds.
        """
        return self._handle.getFPGAWallclock()

    def get_hvps_tuning_info(self) -> int:
        """
        Returns HVPS tuning info (unix format timestamp, i.e. number of seconds, if previously tuned).
        """
        return self._handle.getHVPSTuningInfo()

    def set_hvps_precision_multiplier(self, multiplier: int):
        """
        Sets HVPS precision multiplier (1,2,4,8)
        :param multiplier: multiplier value
        """
        self._handle.setHVPSPrecisionMultiplier(multiplier)


