import arrus.ops.us4r
import numpy as np
import arrus.ops
import dataclasses


@dataclasses.dataclass(frozen=True)
class LinSequence(arrus.ops.Operation):
    """
    A sequence of Tx/Rx operations for classical beamforming (linear scanning).

    :param tx_aperture_center_element: vector of tx aperture center elements [element]
    :param tx_aperture_size: size of the tx aperture [element]
    :param rx_aperture_center_element: vector of rx aperture center elements [element]
    :param rx_aperture_size: size of the rx aperture [element]
    :param tx_focus: tx focal length [m]
    :param pulse: an excitation to perform
    :param pri: pulse repetition interval [s]
    :param downsampling_factor: downsampling factor (decimation), integer factor for decreasing sampling frequency of the output signal
    :param speed_of_sound: assumed speed of sound; can be None, in this case a medium in current context will be used to determine speed of sound [m/s]
    :param tgc_start: tgc starting gain [dB]
    :param tgc_slope: tgc gain slope [dB/m]
    """
    tx_aperture_center_element: np.ndarray
    tx_aperture_size: float
    tx_focus: float
    pulse: arrus.ops.us4r.Pulse
    rx_aperture_center_element: np.ndarray
    rx_aperture_size: float
    pri: float
    rx_sample_range: tuple
    tgc_start: float
    tgc_slope: float
    speed_of_sound: float = None
    downsampling_factor: int = 1
