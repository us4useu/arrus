from dataclasses import dataclass
import typing
import numpy as np
from arrus.ops.operation import Operation


@dataclass(frozen=True)
class Pulse:
    """
    A definition of the pulse that can be triggered by the us4r device.
    """
    center_frequency: float
    n_periods: float
    inverse: bool


@dataclass(frozen=True)
class Tx(Operation):
    """
    Single atomic operation of a signal transmit.

    :param aperture: a set of TX channels that should be enabled
    :param pulse: an excitation to perform
    :param delays: an array of delays to set to active elements. Should have the \
        shape (n_a,), where n_a is a number of active elements determined by \
        tx aperture. When None, firings are performed with no delay (delays=0) [s].
    """
    aperture: np.ndarray
    excitation: Pulse
    delays: typing.Optional[np.ndarray] = None

    def __post_init__(self):
        if self.delays is not None and len(self.delays.shape) != 1:
            raise ValueError("The array of delays should be a vector of "
                             "shape (number of active elements,)")
        if self.delays is not None \
                and self.delays.shape[0] != np.sum(self.aperture):
            raise ValueError(f"The array of delays should have the size equal "
                             f"to the number of active elements of aperture "
                             f"({self.aperture.shape})")


@dataclass(frozen=True)
class Rx(Operation):
    """
    Single atomic operation of echo data reception.

    :param aperture: a set of RX channels that should be enabled
    :param sample_range: a range of samples to acquire [start, end), starts from 0
    :param downsampling_factor: a sampling frequency divider. For example, if \
        nominal sampling frequency (fs) is equal to 65e6 Hz, ``fs_divider=1``,\
        means to use the nominal fs, ``fs_divider=2`` means to use 32.5e6 Hz, \
        etc.
    :param padding: a pair of values (left, right); the left/right value means
        how many zero-channels should be added from the left/right side of the
        aperture. This parameter helps achieve a regular ndarray when
        a sequence of Rxs has a non-constant aperture size (e.g. classical
        beamforming).
    """
    aperture: np.ndarray
    sample_range: tuple
    downsampling_factor: int = 1
    padding: tuple = (0, 0)

    def get_n_samples(self):
        start, end = self.sample_range
        return end-start


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
    """
    ops: typing.List[TxRx]
    tgc_curve: np.ndarray





