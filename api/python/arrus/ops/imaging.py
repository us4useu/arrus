import arrus.ops.us4r
import numpy as np
import arrus.ops
import dataclasses
from collections.abc import Collection


def assert_is_scalar(key, value):
    if isinstance(value, Collection):
        raise ValueError(f"Parameter {key} should be a scalar.")


@dataclasses.dataclass(frozen=True)
class SimpleTxRxSequence:
    """
    A base class for Lin, PWI and STA sequences.

    At most one of the following needs to be specified:

    - tx_aperture_center_element and tx_aperture_center,
    - rx_aperture_center_element and rx_aperture_center.

    All the following parameters should have the same length
    (if they are not scalar):

    - tx_aperture_center_element/tx_aperture_center,
    - rx_aperture_center_element/rx_aperture_center,
    - tx_aperture_size,
    - angles,
    _ tx_focus.

    If any of the above parameters is a scalar, its value will be
    broadcasted to the size of the largest array.

    All probe or aperture element numeration starts from 0.

    :param tx_aperture_center_element: a list of TX aperture center positions \
      (ordinal number of element). Optional, by default the center of the     \
      probe will be used.
    :param tx_aperture_center:  a list of TX aperture center positions. \
      Optional, by default the center of the probe will be used. [m]
    :param tx_aperture_size: number of elements in TX aperture, by default \
      all probe elements are used [m].
    :param rx_aperture_center_element: a list of RX aperture center positions \
      (ordinal number of element). Optional, by default the center of the     \
      probe will be used.
    :param rx_aperture_center:  a list of RX aperture center positions. \
      Optional, by default the center of the probe will be used. [m]
    :param rx_aperture_size: number of elements in RX aperture, by default \
      all probe elements are used. Has to be scalar.
    :param angles: transmission angles [rad]
    :param tx_focus: transmission focus, must be a scalar (a single value for all TX/RXs) [m]
    :param pulse: a pulse excitation to perform
    :param pri: pulse repetition interval [s]
    :param downsampling_factor: downsampling factor (decimation), integer \
      factor to decrease sampling frequency of the output signal, by default 1.
    :param speed_of_sound: assumed speed of sound [m/s]
    :param tgc_start: tgc starting gain [dB]
    :param tgc_slope: tgc gain slope [dB/m]
    :param sri: sequence repetition interval - the time between consecutive RF \
      frames. When None, the time between consecutive RF frames is determined \
      by the total pri only. [s]
    :param n_repeats: size of a single batch -- how many times this sequence should be \
      repeated before data is transferred to computer (integer)
    """
    pulse: arrus.ops.us4r.Pulse
    rx_sample_range: tuple
    pri: float
    sri: float = None
    speed_of_sound: float = None
    tx_focus: object = np.inf
    angles: object = 0.0
    downsampling_factor: int = 1
    tx_aperture_center_element: list = None
    tx_aperture_center: list = None
    tx_aperture_size: int = None
    rx_aperture_center_element: list = None
    rx_aperture_center: list = None
    rx_aperture_size: int = None
    tgc_start: float = None
    tgc_slope: float = None
    tgc_curve: list = None
    n_repeats: int = 1

    def __post_init__(self):
        # Validation
        self.__assert_at_most_one(
            tx_aperture_center_element=self.tx_aperture_center_element,
            tx_aperture_center=self.tx_aperture_center)
        self.__assert_at_most_one(
            rx_aperture_center_element=self.rx_aperture_center_element,
            rx_aperture_center=self.rx_aperture_center)

        self.__assert_at_most_one(
            tgc_start=self.tgc_start,
            tgc_curve=self.tgc_curve)
        self.__assert_at_most_one(
            tgc_start=self.tgc_slope,
            tgc_curve=self.tgc_curve)

        if self.downsampling_factor < 1:
            raise ValueError("Downsampling factor should be a positive value.")

        assert_is_scalar("rx_aperture_size", self.rx_aperture_size)
        assert_is_scalar("tx_focus", self.tx_focus)

        # All the parameters should be 1D.
        self.__assert_is_1d(
            tx_focus=self.tx_focus,
            angles=self.angles,
            tx_aperture_center_element=self.tx_aperture_center_element,
            tx_aperture_center=self.tx_aperture_center,
            tx_aperture_size=self.tx_aperture_size,
            rx_aperture_center_element=self.rx_aperture_center_element,
            rx_aperture_center=self.rx_aperture_center,
            tgc_curve=self.tgc_curve)
        # All given parameters should have the same length (or to be a scalar)
        self.__assert_same_length_lists(
            tx_focus=self.tx_focus,
            angles=self.angles,
            tx_aperture_center_element=self.tx_aperture_center_element,
            tx_aperture_center=self.tx_aperture_center,
            tx_aperture_size=self.tx_aperture_size,
            rx_aperture_center_element=self.rx_aperture_center_element,
            rx_aperture_center=self.rx_aperture_center)

    def __assert_at_most_one(self, **kwargs):
        keys, values = list(zip(*kwargs.items()))
        non_none_values = [value for value in values if value is not None]
        if len(non_none_values) > 1:
            raise ValueError(
                f"At most one of the following can be specified: {keys}")

    def __assert_is_1d(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray) and len(v.shape) > 1:
                raise ValueError(f"Parameter {k} should 1D array.")

    def __assert_same_length_lists(self, **kwargs):
        list_inputs = ((k, v) for k, v in kwargs.items() if isinstance(v, Collection))
        list_inputs = ((len(v), k) for k, v in list_inputs if len(v) > 1)
        dict_inputs = {}
        for l, k in list_inputs:
            dict_inputs.setdefault(l, []).append(k)
        if len(dict_inputs) > 1:
            msg = ",".join([f"{keys}: length: {l}" for l, keys in dict_inputs.items()])
            raise ValueError("All TX/RX parameters that are given as lists "
                             "should have the same length."
                             f"Found: {msg}")


@dataclasses.dataclass(frozen=True)
class LinSequence(SimpleTxRxSequence):
    """
    Linear scanning (classical beamforming) TX/RX sequence.

    Requirements:

    - tx_focus must be finite positive and scalar,
    - tx angle must be a scalar.

    :param init_delay: when the record should start, \
      available options: 'tx_start' - the first recorded sample is when the  \
      transmit starts, 'tx_center' - the first recorded sample is delayed by \
      tx aperture center delay and burst factor.
    """
    init_delay: str = "tx_start"

    def __post_init__(self):
        super().__post_init__()
        if self.tx_focus <= 0 or np.isinf(self.tx_focus):
            raise ValueError("TX focus has to be a positive value.")


@dataclasses.dataclass(frozen=True)
class PwiSequence(SimpleTxRxSequence):
    """
    A sequence of Tx/Rx operations for plane wave imaging.

    Requirements:
    - tx_focus has to be infinity (default value).
    """

    def __post_init__(self):
        super().__post_init__()
        if self.tx_focus != np.inf:
            raise ValueError("TX focus has to be inf (use default value).")


@dataclasses.dataclass(frozen=True)
class StaSequence(SimpleTxRxSequence):
    """
    A sequence of Tx/Rx operations for synthetic transmit aperture
    (diverging beams).

    Requirements:
    - tx focus has to be finite, non-positive.
    """
    def __post_init__(self):
        super().__post_init__()
        if self.tx_focus > 0 or np.isinf(self.tx_focus):
            raise ValueError("TX focus has to be a non-positive value.")
