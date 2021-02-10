import arrus.ops.us4r
import numpy as np
import arrus.ops
import dataclasses


@dataclasses.dataclass(frozen=True)
class LinSequence(arrus.ops.Operation):
    """
    A sequence of Tx/Rx operations for classical beamforming (linear scanning).

    Tx/Rxs from a given sequence differ in the location of the tx/rx aperture
    center (you can provide array of tx or rx aperture center elements).

    :param tx_aperture_center_element: vector of tx aperture center elements \
      [element]
    :param tx_aperture_size: size of the tx aperture [element]
    :param rx_aperture_center_element: vector of rx aperture center elements \
      [element]
    :param rx_aperture_size: size of the rx aperture [element]
    :param tx_focus: tx focal length [m]
    :param pulse: an excitation to perform
    :param rx_sample_range: [start, end) sample; total number of samples should be divisible by 64
    :param pri: pulse repetition interval [s]
    :param downsampling_factor: downsampling factor (decimation), integer \
      factor for decreasing sampling frequency of the output signal
    :param speed_of_sound: assumed speed of sound; can be None, in this case \
      a medium in current context will be used to determine speed of sound [m/s]
    :param tgc_start: tgc starting gain [dB]
    :param tgc_slope: tgc gain slope [dB/m]
    :param sri: sequence repetition interval - the time between consecutive \
      RF frames. When None, the time between consecutive RF frames is \
      determined by the total pri only. [s]
    :param init_delay: when the record should start, available options: 'tx_start' - the first recorded sample is when the transmit starts, 'tx_center' - the first recorded sample is delayed by tx aperture center delay and burst factor
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
    sri: float = None
    init_delay: str = "tx_start"



@dataclasses.dataclass(frozen=True)
class PwiSequence:
    """
    A sequence of Tx/Rx operations for plane wave imaging.

    Tx/Rxs from a given sequence differ in the plane wave angle
    (you can provide a list of angles to transmit).

    Currently, full tx/rx aperture is used.

    :param angles: transmission angles [rad]
    :param pulse: an excitation to perform
    :param pri: pulse repetition interval [s]
    :param downsampling_factor: downsampling factor (decimation), integer
      factor for decreasing sampling frequency of the output signal
    :param speed_of_sound: assumed speed of sound [m/s]
    :param tgc_start: tgc starting gain [dB]
    :param tgc_slope: tgc gain slope [dB/m]
    :param sri: sequence repetition interval - the time between consecutive RF
      frames. When None, the time between consecutive RF frames is determined
      by the total pri only. [s]
    """
    angles: list
    pulse: arrus.ops.us4r.Pulse
    rx_sample_range: tuple
    downsampling_factor: int
    speed_of_sound: float
    pri: float
    tgc_start: float
    tgc_slope: float
    sri: float


@dataclasses.dataclass(frozen=True)
class StaSequence:
    """
    A sequence of Tx/Rx operations for synthetic transmit aperture.

    Currently, a single-element transmit is supported only.

    :param tx_aperture_center_element: vector of tx aperture center elements \
      [element]
    :param rx_aperture_center_element: vector of rx aperture center elements \
      [element]
    :param rx_aperture_size: size of the rx aperture [element]
    :param angles: transmission angles [rad]
    :param pulse: an excitation to perform
    :param pri: pulse repetition interval [s]
    :param downsampling_factor: downsampling factor (decimation), integer
      factor for decreasing sampling frequency of the output signal
    :param speed_of_sound: assumed speed of sound [m/s]
    :param tgc_start: tgc starting gain [dB]
    :param tgc_slope: tgc gain slope [dB/m]
    :param sri: sequence repetition interval - the time between consecutive RF
      frames. When None, the time between consecutive RF frames is determined
      by the total pri only. [s]
    """
    pulse: arrus.ops.us4r.Pulse
    tx_aperture_center_element: list
    rx_aperture_center_element: int
    rx_aperture_size: int
    rx_sample_range: tuple
    downsampling_factor: int
    speed_of_sound: float
    pri: float
    tgc_start: float
    tgc_slope: float
    sri: float
