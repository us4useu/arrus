from dataclasses import dataclass
from collections.abc import Iterable
from abc import ABC
import typing
import numpy as np

import arrus.params

import arrus


class Operation(ABC):
    """
    A single operation to perform.

    This class is abstract and should not be instantiated.
    """
    pass


@dataclass(frozen=True)
class Tx(Operation):
    """
    Single atomic operation of signal transmit.

    :var delays: an array of delays to set to active elements. Should have the \
        shape (n_a,), where n_a is a number of active elements determined by \
        tx aperture. When None, firings are performed with not delay (delays=0).
    :var excitation: an excitation to perform
    :var aperture: a set of TX channels that should be enabled
    :var pri: pulse repetition interval
    """
    excitation: arrus.params.Excitation
    aperture: arrus.params.Aperture
    pri: float
    delays: typing.Optional[np.ndarray] = None

    def __post_init__(self):
        if len(self.delays.shape) != 1:
            raise ValueError("The array of delays should be a vector of "
                             "shape (number of active elements,)")
        if self.delays.shape[0] != self.aperture.get_size():
            raise ValueError(f"The array of delays should have the size equal "
                             f"to the number of active elements of aperture "
                             f"({self.aperture.get_size()})")


@dataclass(frozen=True)
class Rx(Operation):
    """
    Single atomic operation of signal data reception.

    :var samples: number of samples to acquire
    :var decimation: degree of decimation to be made during Rx phase. For \
        example, setting decimation to ``2`` means saving only the half of the \
        acquired data to the device's memory (e.g. this allows to get sampling \
        rate of the acquired data equal to 32.5e6 Hz, \
        assuming that system's sampling rate is equal 65e6 Hz). \
        Check ``get_actual_n_samples``to get the actual number of acquired \
        samples.
    :var aperture: a set of RX channels that should be enabled
    :var rx_time: the total acquisition time
    :var rx_delay: initial rx delay
    """
    n_samples: int
    aperture: arrus.params.Aperture
    decimation: int = 1
    rx_time: float = 160e-6
    rx_delay: float = 5e-6

    def get_actual_n_samples(self):
        return self.n_samples//self.decimation

@dataclass(frozen=True)
class TxRx(Operation):
    """
    Single atomic operation of pulse transmit and signal data reception.
    Returns a result signal data acquired during the sequence.

    :var tx: signal transmit to perform
    :var rx: signal reception to perform
    """
    tx: Tx
    rx: Rx


@dataclass(frozen=True)
class Sequence(Operation):
    """
    A sequence of operations to perform. Returns a result signal data acquired
    during the sequence.

    :var operations: sequence of TX/RX operations to perform
    """
    operations: typing.List[TxRx]


@dataclass(frozen=True)
class Loop(Operation):
    """
    Performs given operation in a loop.

    This operation returns:

    - no value, when callback function is provided
    - a data queue, when no callback function is provided

    :var operation: an operation to perform in a loop
    """
    operation: Operation


@dataclass(frozen=True)
class SetHVVoltage(Operation):
    """
    Sets voltage on a given device. Returns no value.

    :var voltage: voltage to set
    """
    voltage: float


@dataclass(frozen=True)
class DisableHVVoltage(Operation):
    """
    Disables High Voltage supplier in the system.
    """
    pass
