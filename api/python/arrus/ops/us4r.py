from dataclasses import dataclass, field
import typing
import numpy as np
from arrus.ops.operation import Operation
from typing import Iterable
from arrus.framework import Constant


@dataclass(frozen=True)
class Pulse:
    """
    A definition of the pulse that can be triggered by the us4r device.

    :param center_frequency: pulse center frequency [Hz]
    :param n_periods: number of periods of the generated pulse, possible values: 0.5, 1, 1.5, ...
    :param inverse: true if the signal amplitude should be reversed, false otherwise
    """
    center_frequency: float
    n_periods: float
    inverse: bool


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
    :param angle: transmission angles [rad]
    :param speed_of_sound: assumed speed of sound [m/s]
    """
    aperture: typing.Union[np.ndarray, Aperture]
    excitation: Pulse
    delays: typing.Optional[np.ndarray] = None
    focus: typing.Optional[float] = None
    angle: typing.Optional[float] = None
    speed_of_sound: typing.Optional[float] = None

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
                raise ValueError(f"The array of delays should have the size equal "
                                 f"to the number of active elements of aperture "
                                 f"({self.aperture.shape})")
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
    """
    aperture: typing.Union[np.ndarray, Aperture]
    sample_range: tuple
    downsampling_factor: int = 1
    padding: tuple = (0, 0)
    init_delay: str = "tx_start"

    def __post_init__(self):
        if not isinstance(self.aperture, Aperture):
            object.__setattr__(self, "aperture", np.asarray(self.aperture))

    def get_n_samples(self):
        start, end = self.sample_range
        return end - start


@dataclass(frozen=True)
class TxRx:
    """
    Single atomic operation of pulse transmit and signal data reception.

    :param tx: signal transmit to perform
    :param rx: signal reception to perform
    :param pri: pulse repetition interval [s] - time to next event
    """
    tx: Tx
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
    ops: typing.List[TxRx]
    tgc_curve: typing.Union[np.ndarray, Iterable] = field(default_factory=lambda: [])
    sri: float = None
    n_repeats: int = 1

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
    :param work_mode: determines the system work mode, available values: 'ASYNC', 'HOST', 'MANUAL'
    :param processing: data processing to perform on the raw channel RF data \
      currently only arrus.utils.imaging is supported
    """
    tx_rx_sequence: TxRxSequence
    rx_buffer_size: int = 2
    output_buffer: DataBufferSpec = DataBufferSpec(type="FIFO", n_elements=4)
    work_mode: str = "HOST"
    processing: object = None
    digital_down_conversion: DigitalDownConversion = None
    constants: typing.List[Constant] = tuple()
