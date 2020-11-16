import numpy as np
import arrus.exceptions
from arrus.ops.us4r import (
    Tx, Rx, TxRx, TxRxSequence, Pulse
)
import arrus.utils.imaging


def create_lin_sequence(context):
    """
    The function creates list of TxRx objects describing classic scheme.

    :param context: KernelExecutionContext object
    """
    # device parameters
    n_elem = context.device.probe.model.n_elements
    pitch = context.device.probe.model.pitch
    # sequence parameters
    op = context.op

    n_elem_sub = op.tx_aperture_size
    focal_depth = op.tx_focus
    sample_range = op.rx_sample_range
    start_sample, end_sample = sample_range
    pulse = op.pulse
    downsampling_factor = op.downsampling_factor
    pri = op.pri
    fs = context.device.sampling_frequency/op.downsampling_factor

    tx_centers = op.tx_aperture_center_element
    tx_ap_size = op.tx_aperture_size
    rx_centers = op.rx_aperture_center_element
    rx_ap_size = op.rx_aperture_size
    if tx_centers.shape != rx_centers.shape:
        raise arrus.exceptions.IllegalArgumentError(
            "Tx and rx aperture center elements list should have the "
            "same length")
    tgc_start = op.tgc_start
    tgc_slope = op.tgc_slope

    # medium parameters
    c = op.speed_of_sound
    if c is None:
        c = context.medium.speed_of_sound

    distance = np.arange(start=round(400/downsampling_factor),
                         stop=end_sample,
                         step=round(150/downsampling_factor))/fs*c

    tgc_curve = tgc_start + distance*tgc_slope

    # tx/rx aperture, tx delays
    if np.mod(n_elem_sub, 2) == 0:
        # Move focal position to the center of the floor(n_sub_elem/2) element
        focus = [-pitch/2, focal_depth]
    else:
        focus = [0, focal_depth]

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

    tx_apertures = []
    for tx_center_element in tx_centers:
        tx_apertures.append(get_ap(tx_center_element, tx_ap_size))

    tx_delays = compute_tx_delays(op, context.device.probe.model, c)
    for
    txrxs = []

    # use tx apertures to compute tx delays

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
    return TxRxSequence(txrxlist, tgc_curve=tgc_curve)


def get_tx_aperture_center_coords(sequence, probe):
    n_elements = probe.n_elements
    pitch = probe.pitch
    curvature_radius = probe.curvature_radius
    tx_aperture_center_element = sequence.tx_aperture_center_element

    element_position = np.arange(-(n_elements - 1) / 2,
                                 (n_elements - 1) / 2)*pitch

    if not probe.is_convex_array():
        angle = np.zeros(n_elements)
    else:
        angle = element_position / curvature_radius

    tx_aperture_center_angle = np.interp(tx_aperture_center_element,
                                         np.arange(1, n_elements), angle)
    tx_aperture_center_z = np.interp(tx_aperture_center_element,
                                     np.arange(1, n_elements),
                                     probe.element_pos_z)
    tx_aperture_center_x = np.interp(tx_aperture_center_element,
                                     np.arange(1, n_elements),
                                     probe.element_pos_x)

    return tx_aperture_center_angle, tx_aperture_center_x, tx_aperture_center_z


def compute_tx_delays(sequence, probe, speed_of_sound):
    element_x, element_z = probe.element_pos_x, probe.element_pos_z
    element_x, element_z = np.atleast_2d(element_x), np.atleast_2d(element_z)

    tx_center_angle, tx_center_x, tx_center_z = get_tx_aperture_center_coords(
        sequence, probe)
    tx_center_angle = np.atleast_2d(tx_center_angle)
    tx_center_x = np.atleast_2d(tx_center_x)
    tx_center_z = np.atleast_2d(tx_center_z)

    tx_angle = 0
    tx_focus = sequence.tx_focus
    tx_angle_cartesian = tx_center_angle + tx_angle

    focus_x = tx_center_x + tx_focus*np.sin(tx_angle_cartesian)
    focus_z = tx_center_z + tx_focus*np.cos(tx_angle_cartesian)

    # (n_elements, n_tx)
    tx_delays = np.sqrt((focus_x - element_x.T)**2 + (focus_z-element_z.T)**2) / speed_of_sound
    tx_delays_center = np.sqrt((focus_x - tx_center_x.T)**2 + (focus_z-tx_center_z.T)**2) / speed_of_sound

    foc_defoc = 1 - 2*float(tx_focus > 0)
    tx_delays = tx_delays*foc_defoc
    tx_delays_center = tx_delays_center*foc_defoc



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

    distance = np.sqrt()

    dist_max = np.amax(element2focus_distance)
    delays = (dist_max - element2focus_distance)/c
    return delays

