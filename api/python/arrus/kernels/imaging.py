from collections.abc import Collection
import numpy as np
import arrus.exceptions
from arrus.ops.us4r import (
    Tx, Rx, TxRx, TxRxSequence, Pulse
)
from arrus.ops.imaging import (
    LinSequence, PwiSequence, StaSequence
)
from arrus.ops.tgc import LinearTgc
import arrus.utils.imaging
import arrus.kernels.tgc


def __get_aperture_with_padding(center_element, size, probe_model):
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


def __get_speed_of_sound(context):
    if context.op.speed_of_sound is not None:
        return context.op.speed_of_sound
    else:
        return context.medium.speed_of_sound


def get_init_delay(pulse, tx_delay_center):
    """
    Returns burst factor [s]
    """
    n_periods = pulse.n_periods
    fc = pulse.center_frequency
    burst_factor = n_periods / (2 * fc)
    delay = burst_factor + tx_delay_center
    return delay


def __get_sample_range(context, tx_delay_center):
    op = context.op
    sample_range = op.rx_sample_range

    if not isinstance(op, LinSequence):
        return sample_range

    init_delay = op.init_delay
    pulse = op.pulse

    if init_delay == "tx_start":
        return sample_range
    elif init_delay == "tx_center":
        fs = context.device.sampling_frequency/op.downsampling_factor
        delay = get_init_delay(pulse, tx_delay_center) # [s]
        delay = delay*fs
        return tuple(int(round(v+delay)) for v in sample_range)
    else:
        raise ValueError(f"Unrecognized value '{init_delay}' for init_delay.")


def __get_tgc_curve(context):
    op = context.op
    tgc_start = op.tgc_start
    tgc_slope = op.tgc_slope

    if tgc_start is None or tgc_slope is None:
        if op.tgc_curve is None:
            return []
        else:
            return op.tgc_curve
    else:
        return arrus.kernels.tgc.compute_linear_tgc(
            context,
            arrus.ops.tgc.LinearTgc(tgc_start, tgc_slope))


def process_simple_tx_rx_sequence(context):
    op = context.op

    # Medium
    c = __get_speed_of_sound(context)

    # TX/RX
    tx_rx_params = compute_tx_rx_params(
        probe=context.device.probe.model, sequence=op, c=c)
    n_tx = len(tx_rx_params["tx_ap_cent"])
    sample_range = __get_sample_range(context, tx_rx_params["tx_center_delay"])

    # TGC
    tgc_curve = __get_tgc_curve(context)

    txrx = []
    for i in range(n_tx):
        tx_aperture = tx_rx_params["tx_apertures"][i]
        tx_delays = tx_rx_params["tx_delays"][i]

        rx_aperture = tx_rx_params["rx_apertures"][i]
        rx_padding = tx_rx_params["rx_paddings"][i]

        tx = Tx(tx_aperture, op.pulse, tx_delays)
        rx = Rx(rx_aperture, sample_range, op.downsampling_factor,
                padding=rx_padding)
        txrx.append(TxRx(tx, rx, op.pri))
    return TxRxSequence(txrx, tgc_curve=tgc_curve, sri=op.sri,
                        n_repeats=op.n_repeats)


def get_aperture_center(tx_aperture_center_element, probe):
    """
    Interpolates given TX aperture center elements into the positions
    in a probe's cooordinate system.
    """
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


def compute_tx_rx_params(probe, sequence, c):
    """
    Computes tx rx delays for provided angle and focus.
    """
    # INPUT
    tx_rxs = preprocess_sequence_parameters(probe, sequence)
    tx_angle = tx_rxs["tx_angle"]
    tx_focus = sequence.tx_focus
    tx_ap_cent = tx_rxs["tx_ap_cent"]
    tx_ap_size = tx_rxs["tx_ap_size"]
    rx_ap_cent = tx_rxs["rx_ap_cent"]
    rx_ap_size = tx_rxs["rx_ap_size"]

    n_tx = len(tx_angle)

    # OUTPUT
    tx_apertures = []
    tx_delays = []
    tx_center_delays = []
    rx_apertures = []
    rx_paddings = []

    # APERTURES
    for tx_center, tx_size, rx_center, rx_size \
            in zip(tx_ap_cent, tx_ap_size, rx_ap_cent, rx_ap_size):
        tx_aperture, _ = __get_aperture_with_padding(tx_center, tx_size, probe)
        rx_aperture, rx_padding = __get_aperture_with_padding(rx_center,
                                                              rx_size, probe)

        tx_apertures.append(tx_aperture)
        rx_apertures.append(rx_aperture)
        rx_paddings.append(rx_padding)

    # DELAYS
    # Probe element positions (x, z)
    tx_center_angles, tx_center_x, tx_center_z = \
        get_aperture_center(tx_ap_cent, probe)

    tx_angle = tx_center_angles+tx_angle
    tx_angle = np.atleast_2d(tx_angle).T  # [n_tx, 1]
    tx_center_x = np.atleast_2d(tx_center_x).T  # [n_tx, 1]
    tx_center_z = np.atleast_2d(tx_center_z).T
    element_x, element_z = probe.element_pos_x, probe.element_pos_z
    element_x, element_z = np.atleast_2d(element_x), np.atleast_2d(element_z) # [1,n_elem]

    if np.isinf(tx_focus):
        delays = (element_x*np.sin(tx_angle)
                  + element_z*np.cos(tx_angle))/c  # [n_tx, n_elem]
        center_delays = (tx_center_x*np.sin(tx_angle)
                         + tx_center_z*np.cos(tx_angle))/c  # [n_tx, 1]
    else:
        focus_x = tx_center_x + tx_focus*np.sin(tx_angle)  # [n_tx, 1]
        focus_z = tx_center_z + tx_focus*np.cos(tx_angle)  # [n_tx, 1]
        delays = np.sqrt((focus_x-element_x)**2
                          + (focus_z-element_z)**2) / c  # [n_tx, n_elem]
        center_delays = np.sqrt((focus_x-tx_center_x)**2
                              + (focus_z-tx_center_z)**2) / c  # [n_tx, 1]
        foc_defoc = 1 - 2*float(tx_focus > 0)
        delays = delays*foc_defoc
        center_delays = center_delays*foc_defoc
        center_delays = np.atleast_1d(np.squeeze(center_delays))

    for i in range(n_tx):
        op_delays = delays[i]
        op_center_delay = center_delays[i]
        tx_aperture = tx_apertures[i]

        # Use only delays for the active elements.
        op_delays = op_delays[tx_aperture]

        # Remove negative values.
        delays_min = np.min(op_delays).item()
        op_delays = op_delays-delays_min
        op_center_delay = op_center_delay-delays_min

        tx_delays.append(op_delays)
        tx_center_delays.append(op_center_delay)

    # The common delay applied for center of each TX aperture
    # So we can use a single TX center delay when RX beamforming the data.
    # The center of transmit will be in the same position for all TX/RXs.
    tx_center_delay = np.max(tx_center_delays)
    # Equalize
    for i in range(n_tx):
        tx_delays[i] = tx_delays[i]-tx_center_delays[i]+tx_center_delay
    tx_rxs["tx_apertures"] = tx_apertures
    tx_rxs["tx_delays"] = tx_delays
    tx_rxs["tx_center_delay"] = tx_center_delay
    tx_rxs["rx_apertures"] = rx_apertures
    tx_rxs["rx_paddings"] = rx_paddings
    return tx_rxs


def preprocess_sequence_parameters(probe_model, sequence):
    # Get default values and element position for the given probe.
    n_elem = probe_model.n_elements
    default_ap_cent = n_elem//2-1
    default_ap_size = n_elem
    element_pos = (np.arange(0, n_elem)-(n_elem-1)/2) * probe_model.pitch

    def with_default(value, default):
        return value if value is not None else default

    # convert tx/rx_aperture_center to the element position
    def get_center_element(aperture_center, aperture_center_element):
        if aperture_center is not None:
            return np.interp(aperture_center, element_pos, np.arange(0, n_elem))
        else:
            return aperture_center_element

    tx_center_element = get_center_element(sequence.tx_aperture_center,
                                           sequence.tx_aperture_center_element)
    rx_center_element = get_center_element(sequence.rx_aperture_center,
                                           sequence.rx_aperture_center_element)

    tx_rxs = {
        "tx_angle": sequence.angles,
        "tx_ap_cent": with_default(tx_center_element, default_ap_cent),
        "tx_ap_size": with_default(sequence.tx_aperture_size, default_ap_size),
        "rx_ap_cent": with_default(rx_center_element, default_ap_cent),
        # the below is redundant, it will be always a scalar
        "rx_ap_size": with_default(sequence.rx_aperture_size, default_ap_size)
    }

    # Broadcast values to the numpy.ndarray vectors, all with the same length.

    # Determine ndarray size.
    sizes = set([len(v) if isinstance(v, Collection) else 1
                 for _, v in tx_rxs.items()])
    if len(sizes) > 2 or (len(sizes) == 2 and 1 not in sizes):
        raise ValueError("All TX/RX parameters should be lists of the same "
                         f"sizes or scalars (found sizes: {sizes})")
    if 1 in sizes and len(sizes) == 2:
        sizes.remove(1)
    dst_size = next(iter(sizes))

    # Do the broadcasting
    for k, v in tx_rxs.items():
        if not isinstance(v, np.ndarray):
            v = np.squeeze(np.array(v))
            v = np.atleast_1d(v)
        if len(v) == 1:
            v = np.repeat(v, dst_size)
        tx_rxs[k] = v

    return tx_rxs
