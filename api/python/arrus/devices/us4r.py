import dataclasses
import numpy as np
import time
import ctypes
import collections.abc

from arrus.devices.device import Device, DeviceId, DeviceType
import arrus.exceptions
import arrus.devices.probe
from arrus.devices.us4oem import Us4OEM
import arrus.metadata
import arrus.kernels
import arrus.kernels.kernel
import arrus.kernels.tgc
import arrus.ops.tgc
from collections.abc import Iterable


DEVICE_TYPE = DeviceType("Us4R")


@dataclasses.dataclass(frozen=True)
class FrameChannelMapping:
    """
    Stores information how to get logical order of the data from
    the physical order provided by the us4r device.

    :param frames: a mapping: (logical frame, logical channel) -> physical frame
    :param channels: a mapping: (logical frame, logical channel) -> physical channel
    :param us4oems: a mapping: (logical frame, logical channel) -> us4OEM number
    :param frame_offsets: frame starting number for each us4OEM available in the system
    :param n_frames: number of frames each us4OEM produces
    :param batch_size: number of sequences in a single batch
    """
    frames: np.ndarray
    channels: np.ndarray
    us4oems: np.ndarray
    frame_offsets: np.ndarray
    n_frames: np.ndarray
    batch_size: int = 1


class Us4R(Device):
    """
    A handle to Us4R device.

    Wraps an access to arrus.core.Us4R object.
    """

    def __init__(self, handle, parent_session):
        super().__init__()
        self._handle = handle
        self._session = parent_session
        self._device_id = DeviceId(DEVICE_TYPE,
                                   self._handle.getDeviceId().getOrdinal())
        # Context for the currently running sequence.
        self._current_sequence_context = None

    def get_device_id(self):
        return self._device_id

    def set_tgc(self, tgc_curve):
        """
        Sets TGC samples for given TGC description.

        :param samples: a given TGC to set.
        """
        if isinstance(tgc_curve, arrus.ops.tgc.LinearTgc):
            if self._current_sequence_context is None:
                raise ValueError("There is no tx/rx sequence currently "
                                 "uploaded.")
            tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
                self._current_sequence_context, tgc_curve)
        elif not isinstance(tgc_curve, Iterable):
            raise ValueError(f"Unrecognized tgc type: {type(tgc_curve)}")
        self._handle.setTgcCurve(list(tgc_curve))

    def set_hv_voltage(self, voltage):
        """
        Enables HV and sets a given voltage.

        :param voltage: voltage to set
        """
        self._handle.setVoltage(voltage)

    def start(self):
        """
        Starts uploaded tx/rx sequence execution.
        """
        self._handle.start()

    def stop(self):
        """
        Stops tx/rx sequence execution.
        """
        self._handle.stop()

    @property
    def sampling_frequency(self):
        """
        Device sampling frequency [Hz].
        """
        return self._handle.getSamplingFrequency()

    @property
    def n_us4oems(self):
        return self._handle.getNumberOfUs4OEMs()

    def get_us4oem(self, ordinal: int):
        return Us4OEM(self._handle.getUs4OEM(ordinal))

    @property
    def firmware_version(self):
        result = {}
        # Us4OEMs
        us4oem_ver = []
        for i in range(self.n_us4oems):
            dev = self.get_us4oem(i)
            ver = {
                "main": f"{dev.get_firmware_version():x}",
                "tx": f"{dev.get_tx_firmware_version():x}"
            }
            us4oem_ver.append(ver)
        result["Us4OEM"] = us4oem_ver
        return result

    def set_kernel_context(self, kernel_context):
        self._current_sequence_context = kernel_context

    def get_probe_model(self):
        """
        Returns probe model description.
        """
        import arrus.utils.core
        return arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(0).getModel())

    def set_test_pattern(self, pattern):
        """
        Sets given test ADC test patter to be run by Us4OEM components.
        """
        test_pattern_core = arrus.utils.core.convert_to_test_pattern(pattern)
        self._handle.setTestPattern(test_pattern_core)

    @property
    def channels_mask(self):
        """
        Returns a list of system channels that are masked in the configuration.
        """
        return self._handle.getChannelsMask()

    def _get_dto(self):
        import arrus.utils.core
        probe_model = arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(0).getModel())
        probe_dto = arrus.devices.probe.ProbeDTO(model=probe_model)
        return Us4RDTO(probe=probe_dto, sampling_frequency=65e6)


# ------------------------------------------ LEGACY MOCK
@dataclasses.dataclass(frozen=True)
class Us4RDTO(arrus.devices.device.UltrasoundDeviceDTO):
    probe: arrus.devices.probe.ProbeDTO
    sampling_frequency: float

    def get_id(self):
        return "Us4R:0"
