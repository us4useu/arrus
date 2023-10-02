import dataclasses
import numpy as np

from arrus.devices.device import Device, DeviceId, DeviceType
from arrus.devices.ultrasound import Ultrasound
import arrus.exceptions
import arrus.devices.probe
import arrus.ops.imaging
import arrus.ops.tgc
from arrus.devices.us4oem import Us4OEM
import arrus.metadata
import arrus.kernels
import arrus.kernels.kernel
import arrus.kernels.tgc
from collections.abc import Iterable
from typing import Optional


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


class Backplane:
    """
    Digital backplane of the us4R device.
    """
    def __init__(self, us4r):
        self._us4r = us4r

    def get_serial_number(self) -> str:
        """
        Returns serial number of the digital backplane.
        """
        return self._us4r._handle.getBackplaneSerialNumber()

    def get_revision(self) -> str:
        """
        Returns revision number of the digital backplane.
        """
        return self._us4r._handle.getBackplaneRevision()


class Us4R(Device, Ultrasound):
    """
    A handle to Us4R device.

    Wraps an access to arrus.core.Us4R object.
    """

    def __init__(self, handle):
        super().__init__()
        self._handle = handle
        self._device_id = DeviceId(DEVICE_TYPE,
                                   self._handle.getDeviceId().getOrdinal())
        # Context for the currently running sequence.
        self._kernel_context = None
        self._backplane = Backplane(self)

    def get_device_id(self):
        return self._device_id

    def set_tgc(self, tgc_curve):
        """
        Sets TGC samples for given TGC description.

        :param samples: a given TGC to set.
        """
        if isinstance(tgc_curve, arrus.ops.tgc.LinearTgc):
            if self._kernel_context is None:
                raise ValueError("There is no tx/rx sequence currently "
                                 "uploaded.")
            tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
                self._kernel_context,
                self.current_sampling_frequency,
                tgc_curve)
        elif not isinstance(tgc_curve, Iterable):
            raise ValueError(f"Unrecognized tgc type: {type(tgc_curve)}")
        # Here, TGC curve is iterable.
        # Check if we have a pair of iterables, or a single iterable
        if len(tgc_curve) == 2 and (
                isinstance(tgc_curve[0], Iterable)
                and isinstance(tgc_curve[1], Iterable)):
            t, y = tgc_curve
            self._handle.setTgcCurve(list(t), list(y), True)
        else:
            # Otherwise, assume list of floats, use by default TGC sampling
            # points.
            self._handle.setTgcCurve([float(v) for v in tgc_curve])

    def set_hv_voltage(self, voltage):
        """
        Enables HV and sets a given voltage.

        :param voltage: voltage to set
        """
        self._handle.setVoltage(voltage)

    def disable_hv(self):
        """
        Turns off HV.
        """
        self._handle.disableHV()

    def get_us4oem(self, ordinal: int):
        """
        Returns a handle to us4OEM with the given number.

        :return: an object of type arrus.devices.us4oem.Us4OEM
        """
        from arrus.devices.us4oem import Us4OEM
        return Us4OEM(self._handle.getUs4OEM(ordinal))

    def get_backplane(self) -> Backplane:
        """
        Returns a handle to us4R digital backplane.
        """
        return self._backplane

    @property
    def sampling_frequency(self):
        """
        Device NOMINAL sampling frequency [Hz].
        """
        return self._handle.getSamplingFrequency()

    @property
    def current_sampling_frequency(self):
        """
        Device current Rx data sampling frequency [Hz]. This value depends on
        the TX/RX and DDC parameters (e.g. decimation factor) uploaded on
        the system.
        """
        return self._handle.getCurrentSamplingFrequency()

    @property
    def n_us4oems(self):
        return self._handle.getNumberOfUs4OEMs()

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

    def set_kernel_context(self, kernel_context):
        self._kernel_context = kernel_context
        # Set TGC curve from the context.
        seq = self._kernel_context.op
        if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
            if seq.tgc_start is not None and seq.tgc_slope is not None:
                # NOTE: the below line has to be called after
                # session.upload call, otherwise an invalid sampling
                # frequency may be used.
                self.set_tgc(arrus.ops.tgc.LinearTgc(
                    start=seq.tgc_start,
                    slope=seq.tgc_slope
                ))
            elif seq.tgc_curve is not None:
                self.set_tgc(seq.tgc_curve)
            else:
                self.set_tgc([])
        else:
            self.set_tgc(seq.tgc_curve)

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

    def set_hpf_corner_frequency(self, frequency: int):
        """
        Enables digital High-Pass Filter and sets a given corner frequency.
        Available corner frequency values (Hz): 4520'000, 2420'000,
        1200'000, 600'000, 300'000, 180'000,
        80'000, 40'000, 20'000.

        :param frequency: corner high-pass filter frequency to set
        """
        self._handle.setHpfCornerFrequency(frequency)

    def set_lna_gain(self, gain: int):
        """
        Sets LNA gain.

        Available: 12, 18, 24 [dB].

        :param gain: gain value to set
        """
        self._handle.setLnaGain(gain)

    def set_pga_gain(self, gain: int):
        """
        Sets PGA gain.

        Available: 24, 30 [dB].

        :param gain: gain value to set
        """
        self._handle.setPgaGain(gain)

    def set_dtgc_attenuation(self, attenuation: Optional[int]):
        """
        Sets DTGC attenuation.

        Available: 0, 6, 12, 18, 24, 30, 36, 42 [dB] or None;
        None turns off DTGC.

        :param attenuation: attenuation value to set
        :return:
        """
        self._handle.setDtgcAttenuation(attenuation)

    def disable_hpf(self):
        """
        Disables digital high-pass filter.
        """
        self._handle.disableHpf()

    def set_afe(self, addr, reg):
        """
        Writes AFE register

        :param addr: register address (8-bit)
        :param k: write value (16-bit)
        """
        self._handle.setAfe(addr, reg)

    def get_afe(self, addr):
        """
        Reads AFE register value

        :param addr: register address (8-bit)
        """
        return self._handle.getAfe(addr)

    def hvps_set_voltage(self, voltage):
        self._handle.hvpsSetVoltage(voltage)


    @property
    def channels_mask(self):
        """
        Returns a list of system channels that are masked in the configuration.
        """
        return self._handle.getChannelsMask()

    def get_dto(self):
        import arrus.utils.core
        probe_model = arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(0).getModel())
        probe_dto = arrus.devices.probe.ProbeDTO(model=probe_model)
        return Us4RDTO(
            probe=probe_dto,
            sampling_frequency=self.sampling_frequency
        )

    def get_data_description(self, upload_result, sequence):
        # Prepare data buffer and constant context metadata
        fcm = arrus.core.getFrameChannelMapping(upload_result)
        fcm_us4oems, fcm_frame, fcm_channel, frame_offsets, n_frames = \
            arrus.utils.core.convert_fcm_to_np_arrays(fcm, self.n_us4oems)
        fcm = arrus.devices.us4r.FrameChannelMapping(
            us4oems=fcm_us4oems,
            frames=fcm_frame,
            channels=fcm_channel,
            frame_offsets=frame_offsets,
            n_frames=n_frames,
            batch_size=sequence.n_repeats)
        return arrus.metadata.EchoDataDescription(
            sampling_frequency=self.current_sampling_frequency,
            custom={"frame_channel_mapping": fcm}
        )


# ------------------------------------------ LEGACY MOCK
@dataclasses.dataclass(frozen=True)
class Us4RDTO:
    probe: arrus.devices.probe.ProbeDTO
    sampling_frequency: float

    def get_id(self):
        return "Us4R:0"
