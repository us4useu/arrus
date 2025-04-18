import dataclasses
import numpy as np

from arrus.devices.device import Device, DeviceId, DeviceType, parse_device_id
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
from typing import Optional, Union, Sequence
from arrus.devices.probe import ProbeDTO

from arrus.kernels.simple_tx_rx_sequence import get_sample_range

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

    SEQUENCE_START_VAR = "/Us4R:0/sequence:0/start"
    SEQUENCE_END_VAR = "/Us4R:0/sequence:0/end"

    def __init__(self, handle):
        super().__init__()
        self._handle = handle
        self._device_id = DeviceId(DEVICE_TYPE,
                                   self._handle.getDeviceId().getOrdinal())
        # Context for the currently running sequence.
        self._tgc_context = None
        self._backplane = Backplane(self)

    def get_device_id(self):
        return self._device_id

    def set_tgc(self, tgc_curve):
        """
        Sets TGC samples for given TGC description.

        :param samples: a given TGC to set.
        """
        if tgc_curve is None:
            self._handle.setTgcCurve([])
            return
        elif isinstance(tgc_curve, arrus.ops.tgc.LinearTgc):
            if self._tgc_context is None:
                raise ValueError("TGC context is currently not set. "
                                 "Make sure a TX/RX sequence is uploaded and "
                                 "a medium was specified. ")
            tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
                self._tgc_context,
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

    def set_hv_voltage(self, *args):
        """
        Enables HV and sets a given voltage(s).

        This method expects a list of integers or a list of pairs of integers
        as input.
        A single integer v means that the voltage should be set t +v and -v.

        Voltage is always expected to be positive number (even for V-).

        Examples:
            set_hv_voltage(10) -- sets -10 +10 on TX amplitude 2.
            set_hv_voltage((5, 6), (10, 11)) -- sets -5 V for TX state -1, +6 V for TX state +1, -10 V for TX state -2, +11 V for TX state +2

        :param voltage: a single value (for amplitude level 0)
            or a list of tuples, where voltage[0] are (minus, plus) V level 1,
            voltage[1] are (minus, plus) V level 2.
        """
        if len(args) == 1:
            arrus.utils.core.assert_hv_voltage_correct(args[0])
            self._handle.setVoltage(args[0])
        else:
            voltages = arrus.utils.core.convert_to_hv_voltages(args)
            self._handle.setVoltage(voltages)

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

    def set_tgc_and_context(self, sequences, medium):
        self._tgc_context, tgc = self._get_unique_tgc_context_and_tgc(
            sequences=sequences,
            medium=medium
        )
        self.set_tgc(tgc)

    def get_probe_model(self, ordinal=0):
        """
        Returns probe model description.
        """
        import arrus.utils.core
        return arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(ordinal).getModel())

    def set_test_pattern(self, pattern):
        """
        Sets given test ADC test pattern to be run by Us4OEM components.
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

    def get_lna_gain(self):
        """
        Returns current LNA gain value.

        :return: LNA gain value [dB]
        """
        return self._handle.getLnaGain()

    def set_lna_gain(self, gain: int):
        """
        Sets LNA gain.

        Available: 12, 18, 24 [dB].

        :param gain: gain value to set
        """
        self._handle.setLnaGain(gain)

    def get_pga_gain(self):
        """
        Returns current PGA gain value.

        :return: PGA gain value [dB]
        """
        return self._handle.getPgaGain()

    def set_pga_gain(self, gain: int):
        """
        Sets PGA gain.

        Available: 24, 30 [dB].

        :param gain: gain value to set
        """
        self._handle.setPgaGain(gain)

    def set_lpf_cutoff(self, frequency: int):
        """
        Sets low pass filter cutoff frequency.

        Available: 10e6, 15e6, 20e6, 30e6, 35e6, 50e6 [Hz].

        :param frequency: frequency value to set
        """
        self._handle.setLpfCutoff(frequency)

    def set_active_termination(self, impedance: Optional[int]):
        """
        Sets the impedance value for active termination

        Available: 50, 100, 200, 400 [Ohm] or None;
        None turns off active termination.

        Args:
            impedance (int): the impedance value to set
        """
        self._handle.setActiveTermination(impedance)

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

    def set_io_bitstream(self, id, levels, periods):
        return self._handle.setIOBitstream(id, list(levels), list(periods))

    @property
    def channels_mask(self):
        """
        Returns a list of system channels that are masked in the configuration, for Probe:0.
        DEPRECATED, use get_channels_mask method.
        """
        return self.get_channels_mask(0)

    def get_channels_mask(self, probe_nr):
        """
        Returns a list of system channels that are masked in the configuration, for Probe:{probe_nr}.
        """
        return self._handle.getChannelsMask(probe_nr)

    def get_actual_frequency(self, frequency: float) -> float:
        """
        Return the system TX frequency that would be actually set for the given
        TX frequency.
        The output frequency depends on the frequency discretization performed
        by the driver.

        :param frequency: input frequency
        :return: the actual frequency that will be set
        """
        return self._handle.getActualFrequency(frequency)

    def set_stop_on_overflow(self, is_stop):
        """
        Set the system to stop when (RX or host) buffer overflow is detected (ASYNC mode only).
        This property is set by default to true.
        """
        return self._handle.setStopOnOverflow(is_stop)

    def get_number_of_probes(self):
        return self._handle.getNumberOfProbes()

    def get_dto(self):
        import arrus.utils.core
        n_probes = self._handle.getNumberOfProbes()
        probes = []
        for i in range(n_probes):
            probe_model = arrus.utils.core.convert_to_py_probe_model(
                core_model=self._handle.getProbe(i).getModel())
            probe_dto = arrus.devices.probe.ProbeDTO(
                device_id=DeviceId(arrus.devices.probe.DEVICE_TYPE, i),
                model=probe_model
            )
            probes.append(probe_dto)

        return Us4RDTO(
            probe=probes,
            sampling_frequency=self.sampling_frequency,
            data_sampling_frequency=self.current_sampling_frequency
        )

    def get_data_description(self, upload_result, sequence, array_id):
        # Prepare data buffer and constant context metadata
        fcm = self._get_fcm(array_id, upload_result, sequence)
        rx_offset = arrus.core.getRxOffset(array_id, upload_result)
        return arrus.metadata.EchoDataDescription(
            sampling_frequency=self.current_sampling_frequency,
            custom={
                "frame_channel_mapping": fcm,
                "rx_offset": rx_offset
            }
        )

    def get_data_description_updated_for_subsequence(self, array_id, upload_result, sequence):
        fcm = self._get_fcm(array_id, upload_result, sequence)
        rx_offset = arrus.core.getRxOffset(array_id, upload_result)
        return arrus.metadata.EchoDataDescription(
            sampling_frequency=self.current_sampling_frequency,
            custom={
                "frame_channel_mapping": fcm,
                "rx_offset": rx_offset
            }
        )

    def set_stop_on_overflow(self, is_stop):
        """
        Set the system to stop when (RX or host) buffer overflow is detected (ASYNC mode only).
        This property is set by default to true.
        """
        return self._handle.setStopOnOverflow(is_stop)

    def set_maximum_pulse_length(self, max_length):
        """
        Sets maximum pulse length that can be set during the TX/RX sequence programming.
        None means to use up to 32 TX cycles.

        :param max_length: maxium pulse length (s) nullopt means to use 32 TX cycles (legacy OEM constraint)
        """
        self._handle.setMaximumPulseLength(max_length)


    def _get_fcm(self, array_id, upload_result, sequence):
        """
        Returns frame channel mapping (FCM) extracted from the given upload result, assuming
        the given sequence is running. The new FCM considers whether a sub-sequence is already
        set.

        :param upload_result: self.upload result
        :param sequence: the current sequence to consider
        :return: new frame channel mapping
        """
        fcm = arrus.core.getFrameChannelMapping(array_id, upload_result)
        fcm_us4oems, fcm_frame, fcm_channel, frame_offsets, n_frames = \
            arrus.utils.core.convert_fcm_to_np_arrays(fcm, self.n_us4oems)
        return arrus.devices.us4r.FrameChannelMapping(
            us4oems=fcm_us4oems,
            frames=fcm_frame,
            channels=fcm_channel,
            frame_offsets=frame_offsets,
            n_frames=n_frames,
            batch_size=sequence.n_repeats)

    def _get_unique_tgc_context_and_tgc(self, sequences, medium):
        # Make sure that every sequence gives us the same TGC curve.
        # For that:
        # every should be the same
        # every end_sample, speed_of_sound should be the same
        tgcs = set()
        tgc_contexts = set()
        for seq in sequences:
            if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
                # Determine TGC context
                c = seq.speed_of_sound
                c = c if c is not None else medium.speed_of_sound
                sample_range = get_sample_range(
                    op=seq,
                    fs=self.current_sampling_frequency,
                    speed_of_sound=c
                )
                tgc_contexts.add(
                    arrus.kernels.tgc.TgcCalculationContext(
                        end_sample=sample_range[1],
                        speed_of_sound=c
                    )
                )
                # Determine TGC.
                if seq.tgc_start is not None and seq.tgc_slope is not None:
                    tgcs.add(arrus.ops.tgc.LinearTgc(
                        start=seq.tgc_start,
                        slope=seq.tgc_slope
                    ))
                else:
                    tgcs.add(seq.tgc_curve)
            elif isinstance(seq, arrus.ops.us4r.TxRxSequence):
                # Make the curve hashable.
                curve = tuple(seq.tgc_curve.tolist())
                tgcs.add(curve)
                sample_range = seq.get_sample_range_unique()
                if medium is None:
                    # No context
                    tgc_contexts.add(None)
                else:
                    c = medium.speed_of_sound
                    tgc_contexts.add(
                        arrus.kernels.tgc.TgcCalculationContext(
                            end_sample=sample_range[1],
                            speed_of_sound=c
                        )
                    )
            else:
                raise ValueError(f"Unsupported type of TX/RX sequence: "
                                 f"{type(seq)}")
        if len(tgc_contexts) != 1:
            raise ValueError("The TGC setting context is not unique. "
                             "Please make sure that all you sequences "
                             "are of the same type,"
                             "acquire the same number of samples and specify "
                             "the same speed of sound. "
                             f"Detected TGC contexts: {tgc_contexts}")
        if len(tgcs) != 1:
            raise ValueError("The TGC curve should be unique for all "
                             "TX/RX sequences. Detected TGC curves: "
                             f"{tgcs}")
        tgc = next(iter(tgcs))
        if isinstance(tgc, Iterable):
            tgc = np.array(tgc)
        return next(iter(tgc_contexts)), tgc


# ------------------------------------------ LEGACY MOCK
@dataclasses.dataclass(frozen=True)
class Us4RDTO:
    """
    Us4R Data Transfer Object.

    :param probe: a probe/collections of probes connected to the Us4R device
    :param sampling_frequency: nominal sampling frequency of the device
    :param data_sampling_frequency: the actual data sampling frequency
    """
    probe: Union[ProbeDTO, Sequence[ProbeDTO]]
    sampling_frequency: float
    data_sampling_frequency: float
    # TODO(0.12.0) make id obligatory (will break previous const metadata)
    device_id: DeviceId = DeviceId(DEVICE_TYPE, 0)

    def get_probe_by_id(self, id: Union[DeviceId, str]) -> ProbeDTO:
        if isinstance(id, str):
            id = parse_device_id(id)
        probes = self.probe
        if not isinstance(probes, Iterable):
            probes = (probes, )
        # NOTE: the number of probes is expected to be relatively small (< 10)
        probes = [p for p in self.probe if p.device_id == id]
        if len(probes) == 0:
            raise ValueError(f"There is no probe with id: {id}")
        if len(probes) > 1:
            raise ValueError(f"Detected multiple probes with id: {id}")
        return probes[0]

