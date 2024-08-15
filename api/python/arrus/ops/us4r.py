from dataclasses import dataclass, field
import typing
import numpy as np
from arrus.ops.operation import Operation
from typing import Iterable
from arrus.framework import Constant
import dataclasses
from typing import Iterable, Dict, Union, List, Callable, Sequence, Optional, Set, Sized
from arrus.devices.device import parse_device_id, DeviceId


@dataclass(frozen=True)
class Pulse:
    """
    A definition of the pulse that can be triggered by the us4r device.

    NOTE: this object is assumed to be hashable.

    :param center_frequency: pulse center frequency [Hz]
    :param n_periods: number of periods of the generated pulse, possible values: 0.5, 1, 1.5, ...
    :param inverse: true if the signal amplitude should be reversed, false otherwise
    :param amplitude_level: TX voltage amplitude level to set
    """
    center_frequency: float
    n_periods: float
    inverse: bool
    amplitude_level: int = 1


@dataclass(frozen=True)
class WaveformSegment:
    """
    A single waveform segment.

    :param duration: 1D vector of float values, each duration[i] defines how long the state[i] should last [seconds]
    :param state: 1D vector of integer values, subsequent states to apply.
    """
    duration: Iterable[float]
    state: Iterable[int]


@dataclass(frozen=True)
class Waveform:
    """
    A complete Tx waveform to be applied.

    :param segments: subsequent segments of the waveform
    :param n_repeats: how many times the segments[i] should be repeated
    """
    segments: Iterable[WaveformSegment]
    n_repeats: np.ndarray

    @classmethod
    def create(cls, duration: Iterable[float], state: Iterable[int]):
        return Waveform(
            segments=[WaveformSegment(duration=duration, state=state)],
            n_repeats=[1]
        )

    def __post_init__(self):
        # Validate.
        if len(self.segments) != len(self.n_repeats):
            raise ValueError("The list segments should have the same length "
                             "as the list of number of repeats")


class WaveformBuilder:
    """
    TX waveform builder.
    """

    def __init__(self):
        self.segments = []
        self.n_repeats = []

    def add(self, duration: Iterable[float], state: Iterable[int], n: int = 1):
        self.segments.append(WaveformSegment(
            duration=duration,
            state=state
        ))
        self.n_repeats.append(n)
        return self

    def build(self) -> Waveform:
        return Waveform(segments=self.segments, n_repeats=self.n_repeats)



@dataclass(frozen=True)
class Aperture:
    """
    A definition of the aperture that can be applied on TX or RX.

    The aperture's center can be specified relative to the center of the probe
    (the parameter 'center') or as a probe element's number ('center_element').
    Note, that the probe element can be also fractional.

    :param center_element: aperture's center element
    :param center: aperture's center
    :param size: size of the aperture, in the number of elements,
       None value means to use all probe elements
    """
    center_element: float = None
    center: float = None
    size: int = None

    def __post_init__(self):
        # Validate.
        if not ((self.center_element is not None) ^ (self.center is not None)):
            raise ValueError("Exactly one of the following parameters "
                             "should be provided: center, center_element.")


@dataclass(frozen=True)
class Tx(Operation):
    """
    Single atomic operation of a signal transmit.

    Users can specify TX delays in one of the following ways:

    1. by providing (focus, angle, speed_of_sound): TX delays will be automatically
       determined based on the provided values and probe parameters.
    2. by providing array of raw TX delays.

    Exactly one of the following parameter combination should be provided
    (otherwise you will get validation error): (focus, angle, speed_of_sound) or delays.

    The transmit focus can be specified by pair (focus, angle), which are polar coordinates
    of the TX focus to be applied. A vector with such coordinates is hooked
    at the center of the aperture. focus = np.inf means plane wave transmission.

    The focus depth can be negative, which means TX form a virtual source located
    above the probe (i.e. diverging beam transmission).

    :param aperture: a set of TX channels that should be enabled - a binary \
        mask, where 1 at location i means that the channel should be turned \
        on, 0 means that the channel should be turned off, or an instance of \
        Aperture class.
    :param excitation: an excitation to perform
    :param delays: an array of delays to set to active elements [s]. Should have the \
        shape (n_a,), where n_a is a number of active elements determined by \
        tx aperture. The stored value is always of type numpy.ndarray.
    :param focus: transmission focus depth [m] np.inf means to transmit plane wave
    :param angle: transmission angle [rad]
    :param speed_of_sound: assumed speed of sound [m/s]
    :param placement: id of the probe that should do perform TX
    """
    aperture: Union[np.ndarray, Aperture]
    excitation: Pulse
    delays: Optional[np.ndarray] = None
    focus: Optional[float] = None
    angle: Optional[float] = None
    speed_of_sound: Optional[float] = None
    placement: str = "Probe:0"

    def __post_init__(self):
        # Validate.
        is_one_of_focus_angle_c = (
                self.focus is not None
                or self.angle is not None
                or self.speed_of_sound is not None
        )
        is_raw_delays = self.delays is not None
        if not (is_one_of_focus_angle_c ^ is_raw_delays):
            raise ValueError("Exactly one of the following combinations "
                             "should be provided: "
                             "{(focus, angle, speed of sound), delays}.")

        if is_one_of_focus_angle_c:
            if not (self.focus is not None
                    and self.angle is not None
                    and self.speed_of_sound is not None):
                raise ValueError("All of the following parameters "
                                 "should be provided: "
                                 "focus, angle, speed of sound.")

        elif is_raw_delays:
            object.__setattr__(self, "delays", np.asarray(self.delays))
            if self.delays is not None and len(self.delays.shape) != 1:
                raise ValueError("The array of delays should be a vector of "
                                 "shape (number of active elements,)")
            if self.delays is not None \
                    and self.delays.shape[0] != np.sum(self.aperture):
                raise ValueError(
                    f"The array of delays should have the size equal "
                    f"to the number of active elements of aperture "
                    f"({np.asarray(self.aperture).shape})")
        if not isinstance(self.aperture, Aperture):
            object.__setattr__(self, "aperture", np.asarray(self.aperture))


@dataclass(frozen=True)
class Rx(Operation):
    """
    Single atomic operation of echo data reception.

    :param aperture: a set of RX channels that should be enabled - a binary
        mask, where 1 at location i means that the channel should be turned on, \
        0 means that the channel should be turned off, or an instance of  \
        Aperture class.
    :param sample_range: a range of samples to acquire [start, end), starts from 0
    :param downsampling_factor: a sampling frequency divider. For example, if \
        nominal sampling frequency (fs) is equal to 65e6 Hz, ``downsampling_factor=1``,\
        means to use the nominal fs, ``downsampling_factor=2`` means to use 32.5e6 Hz, \
        etc.
    :param padding: a pair of values (left, right); the left/right value means
        how many zero-channels should be added from the left/right side of the
        aperture. This parameter helps achieve a regular ndarray when
        a sequence of Rxs has a non-constant aperture size (e.g. classical
        beamforming).
    :param init_delay: when the recording should start, \
      available options: 'tx_start' - the first recorded sample is when the  \
      transmit starts, 'tx_center' - the first recorded sample is delayed by \
      tx aperture center delay and burst factor.
    :param placement: id of the probe that should do this RX
    """
    aperture: Union[np.ndarray, Aperture]
    sample_range: tuple
    downsampling_factor: int = 1
    padding: tuple = (0, 0)
    init_delay: str = "tx_start"
    placement: str = "Probe:0"

    def __post_init__(self):
        if not isinstance(self.aperture, Aperture):
            object.__setattr__(self, "aperture", np.asarray(self.aperture))

    def get_n_samples(self):
        start, end = self.sample_range
        return end - start

    def is_nop(self):
        if isinstance(self.aperture, Aperture):
            return self.aperture.size == 0
        else:
            return np.sum(self.aperture) == 0


@dataclass(frozen=True)
class TxRx:
    """
    Single atomic operation of pulse transmit and signal data reception.

    :param tx: signal transmit to perform
    :param rx: signal reception to perform
    :param pri: pulse repetition interval [s] - time to next event
    """
    tx: Union[Tx, Set[Tx]]
    rx: Rx
    pri: float


@dataclass(frozen=True)
class TxRxSequence:
    """
    A sequence of tx/rx operations to perform.

    :param operations: sequence of TX/RX operations to perform
    :param tgc_curve: TGC curve samples [dB]
    :param sri: sequence repetition interval - the time between consecutive RF \
        frames. When None, the time between consecutive RF frames is \
        determined by the total pri only. [s]
    """
    ops: List[TxRx]
    tgc_curve: Union[np.ndarray, Iterable] = field(
        default_factory=lambda: [])
    sri: float = None
    n_repeats: int = 1
    name: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "tgc_curve", np.asarray(self.tgc_curve))

    def get_n_samples(self):
        """
        Returns a set of number of samples that the Tx/Rx sequence defines.
        """
        return {op.rx.get_n_samples() for op in self.ops}

    def get_sample_range(self):
        """
        Returns a set of sample ranges that the Tx/Rx sequence defines.
        """
        return {op.rx.sample_range for op in self.ops}

    def get_sample_range_unique(self):
        """
        Returns a unique sample range that the Tx/Rx sequence defines.
        If there are couple of different number of samples in a single sequence,
        a ValueError will be raised.
        """
        sample_range = self.get_sample_range()
        if len(sample_range) > 1:
            raise ValueError("All TX/RXs should acquire the same sample range.")
        return next(iter(sample_range))

    def get_tx_probe_id_unique(self) -> DeviceId:
        tx_probe_ids = set()
        for op in self.ops:
            txs = op.tx
            if not isinstance(txs, Iterable):
                txs = [txs]
            for tx in txs:
                tx_probe_ids.add(parse_device_id(tx.placement))

        if(len(tx_probe_ids)) > 1:
            raise ValueError(f"All TX/Rxs within this sequence: {self.name} "
                             f"are expected to use the same TX probe, found: "
                             f"{tx_probe_ids}")
        return next(iter(tx_probe_ids))

    def get_rx_probe_id_unique(self) -> DeviceId:
        rx_probe_ids = {parse_device_id(op.rx.placement) for op in self.ops}
        if(len(rx_probe_ids)) > 1:
            raise ValueError(f"All TX/Rxs within this sequence: {self.name} "
                             f"are expected to use the same RX probe, found: "
                             f"{rx_probe_ids}")
        return next(iter(rx_probe_ids))

    def get_subsequence(self, start, end):
        """
        Limits the sequence to the given sub-sequence [start, end] both inclusive.
        """
        return dataclasses.replace(
            self,
            ops=self.ops[start:(end+1)])

@dataclass(frozen=True)
class DigitalDownConversion:
    demodulation_frequency: float
    fir_coefficients: Iterable[float]
    decimation_factor: float


@dataclass(frozen=True)
class DataBufferSpec:
    """
    Output data buffer specification.

    :param n_elements: number of elements the buffer should consists of
    :param type: type of a buffer, available values: "FIFO"
    """
    n_elements: int
    type: str


@dataclass(frozen=True)
class Scheme:
    """
    A scheme to load on the us4r device.

    :param tx_rx_sequence: a sequence of tx/rx parameters to perform
    :param rx_buffer_size: number of elements the rx buffer (allocated on \
      us4r ddr internal memory) should consists of
    :param output_buffer: specification of the output buffer
    :param work_mode: determines the system work mode, available values: 'ASYNC', 'HOST', 'MANUAL', 'MANUAL_OP'
    :param processing: data processing to perform on the raw channel RF data \
      currently only arrus.utils.imaging is supported
    """
    tx_rx_sequence: Union[TxRxSequence, Sequence[TxRxSequence]]
    rx_buffer_size: int = 2
    output_buffer: DataBufferSpec = DataBufferSpec(type="FIFO", n_elements=4)
    work_mode: str = "HOST"
    processing: Union[Callable, Sequence[Callable]] = None
    digital_down_conversion: DigitalDownConversion = None
    constants: List = tuple()
