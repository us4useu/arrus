import arrus.ops.us4r
import numpy as np
import arrus.ops

@dataclass(frozen=True)
class LinSequence(arrus.ops.Operation):
    """
    A sequence of Tx/Rx operations for classical beamforming (linear scanning).

    :param tx_aperture_center_element: vector of tx aperture center elements [element]
    :param tx_aperture_size: size of the tx aperture [element]
    :param rx_aperture_center_element: vector of rx aperture center elements [element]
    :param rx_aperture_size: size of the rx aperture [element]
    :param tx_focus: tx focal length [m]
    :param tx_angle: tx angle [rad]
    :param pulse: an excitation to perform
    :param sampling_frequency: a sampling frequency to apply
      currently can be equal 65e6/n (n=1,2,...) only [Hz]
    :param pri: pulse repetition interval
    :param downsampling_factor: downsampling factor (decimation) , integer factor for decreasing sampling frequency of the output signal
    """
    tx_aperture_center_element: np.ndarray
    tx_aperture_size: float
    tx_focus: float
    tx_angle: float
    pulse: arrus.ops.us4r.Pulse
    rx_aperture_center_element: np.ndarray
    rx_aperture_size: float
    sampling_frequency: float
    pri: float
    downsampling_factor: int