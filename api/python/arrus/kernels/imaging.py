import numpy as np
import arrus.exceptions
from arrus.ops.us4r import (
    Tx, Rx, TxRx, TxRxSequence, Pulse
)
from arrus.ops.tgc import LinearTgc
import arrus.utils.imaging
import arrus.kernels.tgc


# -- STA - synthetic transmit aperture
def create_sta_sequence(context):
    # device parameters
    n_elem = context.device.probe.model.n_elements
    pitch = context.device.probe.model.pitch
    # sequence parameters
    op = context.op
    if not isinstance(op, arrus.ops.imaging.StaSequence):
        raise ValueError("This kernel is intended for Pwi sequence only.")
    focus = None
    # sample_range = op.rx_sample_range
    sample_range = op.rx_sample_range
    pulse = op.pulse
    downsampling_factor = op.downsampling_factor
    pri = op.pri
    sri = op.sri
    fs = context.device.sampling_frequency/downsampling_factor
    tgc_start = op.tgc_start
    tgc_slope = op.tgc_slope
    # medium parameters
    c = op.speed_of_sound
    if c is None:
        c = context.medium.speed_of_sound

    tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
        context,
        arrus.ops.tgc.LinearTgc(tgc_start, tgc_slope))

    rx_cent_elem = op.rx_aperture_center_element
    rx_ap_size = op.rx_aperture_size
    # First active element
    l = max(0, round(rx_cent_elem - (rx_ap_size-1)//2))
    # Last active element
    r = min(n_elem-1, l+rx_ap_size)
    rx_aperture = np.zeros(n_elem, dtype=bool)
    rx_aperture[l:(r+1)] = True

    tx_ap_cent_el = op.tx_aperture_center_element
    tx_ap_cent_el = np.array(tx_ap_cent_el)

    tx_delays = np.zeros(1, dtype=np.float32)

    txrx = []
    for i in tx_ap_cent_el:
        tx_aperture = np.zeros(n_elem, dtype=bool)
        tx_aperture[i] = True
        tx = Tx(tx_aperture, pulse, tx_delays)
        rx = Rx(rx_aperture, sample_range, downsampling_factor)
        txrx.append(TxRx(tx, rx, pri))
    return TxRxSequence(txrx, tgc_curve=tgc_curve, sri=sri)


# -- LIN - classical imaging (scanline by scanline).
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
    sri = op.sri
    fs = context.device.sampling_frequency/op.downsampling_factor
    init_delay = op.init_delay

    tx_centers = np.array(op.tx_aperture_center_element)
    tx_ap_size = op.tx_aperture_size
    rx_centers = np.array(op.rx_aperture_center_element)
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

    tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
        context,
        arrus.ops.tgc.LinearTgc(op.tgc_start, op.tgc_slope))

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
        aperture = np.zeros((n_elem, ), dtype=bool)
        aperture[actual_origin:(actual_end+1)] = True
        return aperture, (left_padding, right_padding)
    tx_apertures, tx_delays, tx_delays_center = compute_tx_parameters(
        op, context.device.probe.model, c)
    txrxs = []

    if init_delay == "tx_start":
        actual_sample_range = sample_range
    elif init_delay == "tx_center":
        n_periods = pulse.n_periods
        fc = pulse.center_frequency
        burst_factor = n_periods / (2 * fc)
        delay = burst_factor + tx_delays_center
        delay = delay*fs
        actual_sample_range = tuple(int(round(v+delay)) for v in sample_range)
    else:
        raise ValueError(f"Unrecognized value '{init_delay}' for init_delay.")

    for tx_aperture, delays, rx_center in zip(tx_apertures, tx_delays,
                                              rx_centers):

        if delays.shape == (1, 1):
            delays = delays.reshape((1, ))
        elif len(delays) == 0:
            delays = np.array([])
        else:
            delays = np.squeeze(delays)

        tx = Tx(tx_aperture, pulse, delays)
        rx_aperture, rx_padding = get_ap(rx_center, rx_ap_size)
        rx = Rx(rx_aperture, actual_sample_range, downsampling_factor,
                rx_padding)
        txrxs.append(TxRx(tx, rx, pri))
    return TxRxSequence(txrxs, tgc_curve=tgc_curve, sri=sri)


def get_aperture_with_padding(center_element, size, probe_model):
    n_elem = probe_model.n_elements
    left_half_size = (size-1)/2  # e.g. size 32 -> 15, size 33 -> 16
    # left side:
    origin = int(round(center_element-left_half_size+1e-9))  # e.g. center 0 -> origin -15
    actual_origin = max(0, origin)
    left_padding = abs(min(origin, 0))  # origin -15 -> left padding 15
    # right side
    # aperture last element, e.g. center 0, size 32 -> 16
    end = origin+size-1
    actual_end = min(n_elem-1, end)
    right_padding = abs(min(actual_end-end, 0))
    aperture = np.zeros((n_elem, ), dtype=bool)
    aperture[int(actual_origin):(int(actual_end)+1)] = True
    return aperture, (left_padding, right_padding)


def get_tx_aperture_center_coords(tx_aperture_center_element, probe):
    n_elements = probe.n_elements
    pitch = probe.pitch
    curvature_radius = probe.curvature_radius

    element_position = np.arange(-(n_elements - 1) / 2,
                                 n_elements / 2)*pitch

    if not probe.is_convex_array():
        angle = np.zeros(n_elements)
    else:
        angle = element_position / curvature_radius


    tx_aperture_center_angle = np.interp(tx_aperture_center_element,
                                         np.arange(0, n_elements), angle)
    tx_aperture_center_z = np.interp(tx_aperture_center_element,
                                     np.arange(0, n_elements),
                                     np.squeeze(probe.element_pos_z))
    tx_aperture_center_x = np.interp(tx_aperture_center_element,
                                     np.arange(0, n_elements),
                                     np.squeeze(probe.element_pos_x))
    return tx_aperture_center_angle, tx_aperture_center_x, tx_aperture_center_z


def compute_tx_parameters(sequence, probe, speed_of_sound):
    tx_ap_size = sequence.tx_aperture_size
    tx_centers = sequence.tx_aperture_center_element
    tx_apertures = []
    for tx_center_element in tx_centers:
        tx_apertures.append(get_aperture_with_padding(tx_center_element,
                                                      tx_ap_size, probe)[0])

    element_x, element_z = probe.element_pos_x, probe.element_pos_z
    element_x, element_z = np.atleast_2d(element_x), np.atleast_2d(element_z)

    tx_center_angle, tx_center_x, tx_center_z = get_tx_aperture_center_coords(
        tx_centers, probe)
    tx_center_angle = np.atleast_2d(tx_center_angle)
    tx_center_x = np.atleast_2d(tx_center_x)
    tx_center_z = np.atleast_2d(tx_center_z)

    tx_angle = 0
    tx_focus = sequence.tx_focus
    tx_angle_cartesian = tx_center_angle + tx_angle

    focus_x = tx_center_x + tx_focus*np.sin(tx_angle_cartesian)
    focus_z = tx_center_z + tx_focus*np.cos(tx_angle_cartesian)

    # (n_elements, n_tx)
    tx_delays = np.sqrt((focus_x-element_x.T)**2
                        + (focus_z-element_z.T)**2) / speed_of_sound
    tx_delays_center = np.sqrt((focus_x-tx_center_x)**2
                               + (focus_z-tx_center_z)**2) / speed_of_sound
    foc_defoc = 1 - 2*float(tx_focus > 0)
    tx_delays = tx_delays*foc_defoc
    tx_delays_center = tx_delays_center*foc_defoc
    tx_delays_center = np.atleast_1d(np.squeeze(tx_delays_center))
    tx_aperture_delays = []
    tx_aperture_delays_center = []

    for i, tx_aperture in enumerate(tx_apertures):
        tx_del = tx_delays[np.argwhere(tx_aperture), i]
        if len(tx_del) > 0:
            tx_delay_shift = - np.min(tx_del)
            tx_del = tx_del + tx_delay_shift
            tx_del_cent = tx_delays_center[i] + tx_delay_shift
        else:
            tx_del = np.array([])
            tx_del_cent = None
        tx_aperture_delays.append(tx_del)
        tx_aperture_delays_center.append(tx_del_cent)

    tx_delays_center_max = np.max(tx_aperture_delays_center)

    # Equalize
    for i in range(len(tx_aperture_delays)):
        if len(tx_aperture_delays[i]) > 0:
            tx_aperture_delays[i] = tx_aperture_delays[i] \
                                  - tx_aperture_delays_center[i] \
                                  + tx_delays_center_max
    return tx_apertures, tx_aperture_delays, tx_delays_center_max


# ---------------------- PWI
def _get_pwi_aperture_description(probe, sequence):
    n_elem = probe.n_elements
    angles = sequence.angles
    angles = np.array(angles)
    n_angles = angles.size
    # TX/RX aperture
    def get_with_default(value, default):
        value = value if value is not None else default
        value = np.squeeze(np.array(value))
        value = np.atleast_1d(value)
        # A scalar value, and we have multiple angles - the value will be
        # broadcasted to given number of angles.
        if len(value) == 1 and n_angles > 1:
            value = np.repeat(value, n_angles)
        return value

    if sequence.tx_aperture_center_element is not None and sequence.tx_aperture_center is not None:
        raise ValueError("Only one of the following can be specified: "
                         "tx_aperture_center_element, tx_aperture_center")
    if sequence.rx_aperture_center_element is not None and sequence.rx_aperture_center is not None:
        raise ValueError("Only one of the following can be specified: "
                         "rx_aperture_center_element, rx_aperture_center")

    default_ap_center = np.repeat(n_elem//2-1, n_angles)
    default_ap_size = np.repeat(n_elem, n_angles)

    # tx/rx_centers - tx/rx center ELEMENTS
    element_pos_along_probe = (np.arange(0, n_elem)-(n_elem-1)/2)*probe.pitch
    if sequence.tx_aperture_center is not None:
        # Tx aperture center is the position ALONG THE PROBE CURVATURE
        tx_centers = np.interp(sequence.tx_aperture_center, element_pos_along_probe, np.arange(0, n_elem))
    else:
        tx_centers = sequence.tx_aperture_center_element
    tx_centers = get_with_default(tx_centers, default_ap_center)

    if sequence.rx_aperture_center is not None:
        rx_centers = np.interp(sequence.rx_aperture_center, element_pos_along_probe, np.arange(0, n_elem))
    else:
        rx_centers = sequence.rx_aperture_center_element
    rx_centers = get_with_default(rx_centers, default_ap_center)

    tx_sizes = get_with_default(sequence.tx_aperture_size, default_ap_size)
    rx_sizes = get_with_default(sequence.rx_aperture_size, default_ap_size)
    return tx_centers, tx_sizes, rx_centers, rx_sizes


# -- PWI - plane wave imaging
def create_pwi_sequence(context):
    # device parameters
    probe_model = context.device.probe.model
    n_elem = probe_model.n_elements
    # sequence parameters
    op = context.op  # PWI Sequence
    if not isinstance(op, arrus.ops.imaging.PwiSequence):
        raise ValueError("This kernel is intended for Pwi sequence only.")
    sample_range = op.rx_sample_range
    pulse = op.pulse
    downsampling_factor = op.downsampling_factor
    pri = op.pri
    sri = op.sri
    angles = op.angles
    angles = np.array(angles)

    tx_centers, tx_sizes, rx_centers, rx_sizes = _get_pwi_aperture_description(
        probe_model, op
    )

    # Validate TX/RX parameters length
    sequence_aperture_params_len = {len(tx_centers), len(rx_centers),
                                 len(tx_sizes), len(rx_sizes)}
    if len(sequence_aperture_params_len) != 1:
        raise ValueError(f"All parameters should have the same length, "
                         f"found:{sequence_aperture_params_len}")

    # medium parameters
    c = op.speed_of_sound
    if c is None:
        c = context.medium.speed_of_sound

    tgc_start = op.tgc_start
    tgc_slope = op.tgc_slope

    if tgc_start is None or tgc_slope is None:
        tgc_curve = op.tgc_curve
    else:
        tgc_curve = arrus.kernels.tgc.compute_linear_tgc(
            context,
            arrus.ops.tgc.LinearTgc(tgc_start, tgc_slope))

    n_angles = np.size(angles)
    tx_apertures, tx_delays, tx_center_delay = _compute_pwi_tx_params(
        probe=context.device.probe.model, sequence=op, c=c)
    txrx = []
    for i_angle in range(n_angles):
        tx_aperture = tx_apertures[i_angle]
        angle_delays = tx_delays[i_angle]

        rx_center = rx_centers[i_angle]
        rx_size = rx_sizes[i_angle]

        rx_aperture, padding = get_aperture_with_padding(rx_center, rx_size, probe_model)

        tx = Tx(tx_aperture, pulse, angle_delays)
        rx = Rx(rx_aperture, sample_range, downsampling_factor, padding=padding)
        txrx.append(TxRx(tx, rx, pri))
    return TxRxSequence(txrx, tgc_curve=tgc_curve, sri=sri)


def _compute_pwi_tx_params(probe, sequence, c):
    """
    Computes tx rx delays for provided angle and focus.

    Currently used by PWI imaging.
    """
    # transducer indexes
    angles = sequence.angles
    tx_centers, tx_sizes, _, _ = _get_pwi_aperture_description(probe, sequence)
    angles = np.squeeze(np.array(angles))
    if angles.size == 0:
        raise ValueError("The list of transmit angles should not be empty.")

    # Probe element positions (x, z)
    element_x, element_z = probe.element_pos_x, probe.element_pos_z
    element_x, element_z = np.atleast_2d(element_x), np.atleast_2d(element_z)

    tx_center_angles, tx_center_x, tx_center_z = \
        get_tx_aperture_center_coords(tx_centers, probe)

    tx_angles = tx_center_angles+angles
    tx_angles = np.atleast_2d(tx_angles).T
    tx_center_x = np.atleast_2d(tx_center_x).T
    tx_center_z = np.atleast_2d(tx_center_z).T

    delays = (element_x*np.sin(tx_angles) + element_z*np.cos(tx_angles))/c
    center_delays = (tx_center_x*np.sin(tx_angles)+tx_center_z*np.cos(tx_angles))/c

    tx_apertures = []
    tx_delays = []
    tx_center_delays = []

    for i in range(angles.size):
        tx_center = tx_centers[i]
        tx_size = tx_sizes[i]
        op_delays = delays[i]
        op_center_delay = center_delays[i]

        tx_aperture, _ = get_aperture_with_padding(tx_center, tx_size, probe)
        # Move to positive values
        op_delays = op_delays[tx_aperture]

        delays_min = np.min(op_delays).item()
        op_delays = op_delays-delays_min
        op_center_delay = op_center_delay-delays_min

        tx_apertures.append(tx_aperture)
        tx_delays.append(op_delays)
        tx_center_delays.append(op_center_delay)

    # The common delay applied for center of each TX aperture
    # So we can use a single TX center delay when beamforming the data
    tx_center_delay = np.max(tx_center_delays)
    # Equalize
    for i in range(angles.size):
        tx_delays[i] = tx_delays[i]-tx_center_delays[i]+tx_center_delay

    return tx_apertures, tx_delays, tx_center_delay
