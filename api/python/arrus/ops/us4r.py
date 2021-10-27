from dataclasses import dataclass
import typing
import numpy as np
from arrus.ops.operation import Operation


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
class Tx(Operation):
    """
    Single atomic operation of a signal transmit.

    :param aperture: a set of TX channels that should be enabled - a binary \
        mask, where 1 at location i means that the channel should be turned \
        on, 0 means that the channel should be turned off
    :param pulse: an excitation to perform
    :param delays: an array of delays to set to active elements. Should have the \
        shape (n_a,), where n_a is a number of active elements determined by \
        tx aperture. When None, firings are performed with no delay (delays=0) [s]\
        The stored value is always of type numpy.ndarray.
    """
    aperture: np.ndarray
    excitation: Pulse
    delays: typing.Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, "delays", np.asarray(self.delays))
        object.__setattr__(self, "aperture", np.asarray(self.aperture))

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

    :param aperture: a set of RX channels that should be enabled - a binary
        mask, where 1 at location i means that the channel should be turned on, \
        0 means that the channel should be turned off. The stored value is \
        always of type numpy.ndarray.
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

    def __post_init__(self):
        object.__setattr__(self, "aperture", np.asarray(self.aperture))

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
    :param sri: sequence repetition interval - the time between consecutive RF \
        frames. When None, the time between consecutive RF frames is \
        determined by the total pri only. [s]
    """
    ops: typing.List[TxRx]
    tgc_curve: np.ndarray
    sri: float = None
    n_repeats: int = 1

    def __post_init__(self):
        object.__setattr__(self, "tgc_curve", np.asarray(self.tgc_curve))

    def get_n_samples(self):
        """
        Returns a set of number of samples that the Tx/Rx sequence defines.
        """
        return {op.rx.get_n_samples() for op in self.ops}


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





