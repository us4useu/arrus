import numpy as np
import arrus.exceptions
from arrus.ops.us4r import (
    Tx, Rx, TxRx, TxRxSequence, Pulse
)


def create_lin_sequence(context):
    """
    The function creates list of TxRx objects describing classic scheme.

    :param context: KernelExecutionContext object
    """
    # device parameters
    n_elem = context.device.probe.n_elements
    pitch = context.device.probe.pitch
    # sequence parameters
    op = context.op
    n_elem_sub = op.tx_aperture_size
    focal_depth = op.tx_focus
    sample_range = op.rx_sample_range
    pulse = op.pulse
    downsampling_factor = op.downsampling_factor
    pri = op.pri
    tx_centers = op.tx_aperture_center_element
    tx_ap_size = op.tx_aperture_size
    rx_centers = op.rx_aperture_center_element
    rx_ap_size = op.rx_aperture_size
    if tx_centers.shape != rx_centers.shape:
        raise arrus.exceptions.IllegalArgumentError(
            "Tx and rx aperture center elements list should have the "
            "same length")

    # medium parameters
    c = context.medium.speed_of_sound

    if np.mod(n_elem_sub, 2) == 0:
        focus = [pitch/2, focal_depth]
    else:
        focus = [0, focal_depth]

    # enumerate delays mask and padding for each tx/rx event
    subaperture_delays = enum_classic_delays(n_elem_sub, pitch, c, focus)

    def get_ap(center_element, size):
        left_half_size = (size-1)//2  # e.g. size 32 -> 15, size 33 -> 16
        right_half_size = size//2  # e.g. size 32 -> 16, size 33 -> 16
        # left side:
        origin = center_element-left_half_size  # e.g. center 0 -> origin -15
        actual_origin = max(0, origin)
        left_padding = abs(min(origin, 0))  # origin -15 -> left padding 15
        # right side
        # aperture last element, e.g. center 0, size 32 -> 16
        end = center_element+right_half_size
        actual_end = min(n_elem-1, end)
        right_padding = abs(min(actual_end-end, 0))
        aperture = np.zeros((n_elem, ), dtype=np.bool)
        aperture[actual_origin:(actual_end+1)] = True
        return aperture, (left_padding, right_padding)

    # create tx/rx objects list
    def create_tx_rx(tx_center_element, rx_center_element):
        # Tx
        tx_aperture, tx_padding = get_ap(tx_center_element, tx_ap_size)
        tx_pad_l, tx_pad_r = tx_padding
        actual_ap_size = tx_ap_size-(tx_pad_l+tx_pad_r)
        tx_delays = np.zeros(actual_ap_size, dtype=np.float32)
        if tx_pad_r > 0:
            tx_delays = subaperture_delays[tx_pad_l:-tx_pad_r]
        else:
            tx_delays = subaperture_delays[tx_pad_l:]
        # Rx
        rx_aperture, rx_padding = get_ap(rx_center_element, rx_ap_size)
        tx = Tx(tx_aperture, pulse, tx_delays)
        rx = Rx(rx_aperture, sample_range, downsampling_factor, rx_padding)
        return TxRx(tx, rx, pri)
    txrxlist = [create_tx_rx(*c) for c in zip(tx_centers, rx_centers)]
    return TxRxSequence(txrxlist, tgc_curve=np.ndarray([]))


def enum_classic_delays(n_elem, pitch, c, focus):
    """
    The function enumerates classical focusing delays for linear array.
    It assumes that the 0 is in the center of the aperture.

    :param n_elem: number of elements in aperture,
    :param pitch: distance between two adjacent probe elements [m],
    :param c: speed of sound [m/s],
    :param focus: coordinates of the focal point ([xf, zf]),
      or focal length only (then xf = 0 is assumed) [m],
    :param delays: output delays vector.
    """
    if np.isscalar(focus):
        xf = 0
        zf = focus
    elif np.shape(focus) == (2,):
        xf, zf = focus
    else:
        raise ValueError("Bad focus - should be scalar, 1-dimensional ndarray, "
                         "or 2-dimensional ndarray")

    aperture_width = (n_elem-1)*pitch
    el_coord_x = np.linspace(-aperture_width/2, aperture_width/2, n_elem)
    element2focus_distance = np.sqrt((el_coord_x - xf)**2 + zf**2)
    dist_max = np.amax(element2focus_distance)
    delays = (dist_max - element2focus_distance)/c
    return delays
