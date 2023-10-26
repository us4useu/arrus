"""This module converts Python TxRxSequence to C++ API TxRxSequence.
In some time this module probably will not be needed anymore,
as all of the below functionality will be moved to C++ API.
"""
import dataclasses
import typing

import numpy as np
from arrus.ops.us4r import (
    TxRxSequence, Tx, Rx, TxRx, Aperture
)
from arrus.kernels.kernel import KernelExecutionContext


def process_tx_rx_sequence(context: KernelExecutionContext):
    """
    Performs all the necessary pre-processing needed to
    run raw TX/RX sequence on the us4r device.

    Currently, this function only translates (focus, angle, speed_of_sound)
    into a list of raw delays.
    """
    sequence: TxRxSequence = context.op
    constants = context.constants
    probe_model = context.device.probe.model
    fs: float = __get_sampling_frequency(context)
    # Update the following sequence parameters (if necessary):
    # - tx: aperture (to binary mask)
    # - rx: aperture (to binary mask), rx padding
    sequence, _ = convert_to_us4r_sequence(
        sequence=sequence,
        probe_model=probe_model,
        fs=fs,
        focus_constants=constants
    )
    return sequence


def convert_to_us4r_sequence(
        sequence: TxRxSequence, probe_model, fs: float, focus_constants=(),
):
    sequence_with_masks: TxRxSequence = set_aperture_masks(
        sequence=sequence,
        probe=probe_model
    )
    # We want all operators to have exactly the same combination
    # of parameters: delays or focus,angle,speed of sound.
    # Currently this is required because of TX center delay equalization
    # (see code below).
    delays_is_none = set(op.tx.delays is None for op in sequence.ops)
    foc_ang_c_is_none = set(
        op.tx.focus is None
        and op.tx.angle is None
        and op.tx.speed_of_sound is None
        for op in sequence.ops
    )
    if len(delays_is_none) != 1 or len(foc_ang_c_is_none) != 1:
        # the above arrays are {True, False}
        raise ValueError("All TX/RXs delays should be None "
                         "or focus,angle,speed of sound should be None.")
    if delays_is_none == {True}:
        # Updates in the input sequence the following parameters:
        # - tx: delays, focus, angle, speed_of_sound
        # - rx: init_delay, sample_range
        # Otherwise, we need to convert focus, angle, c to raw TX delays.

        # Currently we support sequence/txFocus:i constants only.
        # sequence/txFocus:i is logically translated to sequence/op:*/txFocus:i
        # and each sequence/op:j/txFocus:i is translated into sequence/op:j/txDelays:i
        # Finally, in the output sequence we should have
        # a sequence of ops with tx_delays: Constant(sequence/op:j/txDelays:i)
        #
        # If the raw TX/RX sequence is set, each tx_delays should be translated
        # to Constant(sequence/op:j/txDelays:i).
        # The list of ALL tx delays should be specified in the constants.
        # Each TX should have tx_delays set to string.
        # Make sure that it is possible to do implicit conversion Constant -> tx_delays.
        # Otherwise we will break Metadata API.
        # is it possible at all? If not, maybe for backward conversion convert
        # each in the raw_sequence before saving it into const_metadata.
        # In the future versions of API Constant will have to have numpy compatible
        # interface.

        # Input: Tx/Rxs with tx.focus = "sequence/op:i/txFocus:j" or value
        # if this is the second option:

        # Currently support one of the possibilities:
        # all TX/RXs have the focus set to the same string ->

        dels, tx_center_delay = get_tx_delays(
            probe_model, sequence, sequence_with_masks,
        )

        # Calculate delays for each constant.


        new_ops = []
        # Update input sequence.
        for i, op in enumerate(sequence_with_masks.ops):
            d = dels[i]
            sample_range = __get_sample_range(
                op=op, tx_delay_center=tx_center_delay, fs=fs)
            # Replace
            old_tx = op.tx
            new_tx = dataclasses.replace(old_tx, delays=np.atleast_1d(d),
                                         focus=None, angle=None,
                                         speed_of_sound=None)
            old_rx = op.rx
            new_rx = dataclasses.replace(old_rx,
                                         sample_range=sample_range,
                                         init_delay="tx_start")
            new_op = dataclasses.replace(op, tx=new_tx, rx=new_rx)
            new_ops.append(new_op)
        sequence = dataclasses.replace(sequence, ops=new_ops)
        return sequence, tx_center_delay
    else:
        return sequence, None


def get_tx_delays(
        probe, sequence: TxRxSequence, seq_with_masks: TxRxSequence,
):
    """
    Returns tx_center_delay = None when all TXs have empty aperture.
    """
    # COMPUTE TX APERTURE CENTERS.
    tx_aperture_center_element = []
    for op in sequence.ops:
        tx = op.tx
        ap_cent_elem = __get_center_element(
            tx.aperture, probe_model=probe
        )
        tx_aperture_center_element.append(ap_cent_elem)
    tx_aperture_center_element = np.asarray(tx_aperture_center_element)
    # COMPUTE TX DELAYS.
    tx_center_angles, tx_center_x, tx_center_z = \
        get_aperture_center(tx_aperture_center_element, probe)
    tx_delays = []
    center_delays = []
    element_x, element_z = probe.element_pos_x, probe.element_pos_z
    element_x, element_z = np.atleast_2d(element_x), np.atleast_2d(element_z)  # [1,n_elem]
    # Compute tx_delays
    for i, op in enumerate(sequence.ops):
        tx = op.tx
        tx_angle = tx.angle + tx_center_angles[i]
        tx_cent_x = tx_center_x[i]
        tx_cent_z = tx_center_z[i]
        c = tx.speed_of_sound
        assert (tx.focus is not None and tx.angle is not None
                and tx.speed_of_sound is not None)
        if np.isinf(tx.focus):
            # PWI
            delays = (element_x * np.sin(tx_angle)
                      + element_z * np.cos(tx_angle)) / c  # [1, n_elem]
            center_delay = (tx_cent_x * np.sin(tx_angle)
                            + tx_cent_z * np.cos(tx_angle)) / c  # scalar
        else:
            # Virtual source/focus
            focus_x = tx_cent_x + tx.focus * np.sin(tx_angle)  # scalar
            focus_z = tx_cent_z + tx.focus * np.cos(tx_angle)  # scalar
            delays = np.sqrt((focus_x - element_x) ** 2
                             + (focus_z - element_z) ** 2) / c  # [1, n_elem]
            center_delay = np.sqrt((focus_x - tx_cent_x) ** 2
                                   + (focus_z - tx_cent_z) ** 2) / c  # scalar
            foc_defoc = 1 - 2 * float(tx.focus > 0)
            delays = delays * foc_defoc
            center_delay = center_delay * foc_defoc
        tx_delays.append(delays)
        center_delays.append(center_delay)
    # Note: n_elem is the total number of elements
    # of the probe.
    tx_delays = np.concatenate(tx_delays, axis=0)  # [n_tx, n_elem]
    center_delays = np.atleast_1d(np.asarray(center_delays))  # [n_tx, ]

    normalized_tx_delays = []
    normalized_tx_center_delays = []
    # Equalize delays to the TX center.
    for i, op in enumerate(sequence.ops):
        op_delays = tx_delays[i]
        op_center_delay = center_delays[i]
        tx_aperture_mask = seq_with_masks.ops[i].tx.aperture
        # Use only delays for the active elements.
        op_delays = op_delays[tx_aperture_mask]
        if len(op_delays) == 0:
            # Empty TX aperture.
            normalized_tx_delays.append(op_delays)
            normalized_tx_center_delays.append(np.nan)
        else:
            # Move tx delays to bias = 0.
            delays_min = np.min(op_delays).item()
            op_delays = op_delays - delays_min
            op_center_delay = op_center_delay - delays_min
            normalized_tx_delays.append(op_delays)
            normalized_tx_center_delays.append(op_center_delay)
    # Equalize
    # The common delay applied for center of each TX aperture
    # So we can use a single TX center delay when RX beamforming the data.
    # The center of transmit will be in the same position for all TX/RXs.
    # Note: in the case when all TXs have empty TX aperture, None should be
    # returned.
    is_all_empty_tx_aperture = np.asarray([
        len(d) == 0 for d in normalized_tx_delays
    ]).all()
    if is_all_empty_tx_aperture:
        tx_center_delay = None
    else:
        tx_center_delay = np.nanmax(normalized_tx_center_delays)
    equalized_tx_delays = []
    for i, op in enumerate(sequence.ops):
        d = normalized_tx_delays[i]
        if len(d) > 0:
            # Non-empty delays.
            d = d - normalized_tx_center_delays[i] + tx_center_delay
        equalized_tx_delays.append(d)
    return equalized_tx_delays, tx_center_delay


def get_aperture_center(tx_aperture_center_element, probe):
    """
    Interpolates given TX aperture center elements into the positions
    in a probe's cooordinate system.
    """
    n_elements = probe.n_elements
    pitch = probe.pitch
    curvature_radius = probe.curvature_radius
    element_position = np.arange(-(n_elements - 1) / 2,
                                 n_elements / 2) * pitch
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


def __get_center_element(aperture, probe_model):
    tx_ap_cent_elem = 0
    if isinstance(aperture, Aperture):
        return __get_aperture_center_element(aperture, probe_model)
    else:
        # aperture should be a numpy.ndarray
        if np.sum(aperture) == 0:
            # For empty TX aperture, we will eventually
            # get an empty array of delays (see the code
            # below, i.e. the part responsible for taking
            # delays for the aperture elements). So the
            # value below should not matter.
            tx_ap_cent_elem = 0
        elif np.sum(aperture) == 1:
            tx_ap_cent_elem = np.argwhere(aperture)
        else:
            ap_elems = np.argwhere(aperture).squeeze()
            loc_diff = set(np.diff(ap_elems))
            if loc_diff != {1}:
                raise ValueError("Continuous TX aperture is required "
                                 "for focus, angle, speed of sound "
                                 "combination.")
            tx_ap_start = np.min(ap_elems)
            tx_ap_end = np.max(ap_elems)
            tx_ap_cent_elem = (tx_ap_start + tx_ap_end) / 2
    return tx_ap_cent_elem


def get_apertures_center_elements(apertures: typing.Iterable[Aperture],
                                  probe_model):
    return np.asarray([
        __get_aperture_center_element(ap, probe_model)
        for ap in apertures
    ])


def get_apertures_sizes(apertures, probe_model):
    return np.asarray([
        ap.size if ap.size is not None else probe_model.n_elements
        for ap in apertures
    ])


def __get_aperture_center_element(aperture: Aperture, probe_model):
    if aperture.center_element is not None:
        return aperture.center_element
    elif aperture.center is not None:
        n_elem = probe_model.n_elements
        element_pos = (np.arange(0, n_elem) - (n_elem - 1) / 2) * probe_model.pitch
        return np.interp(aperture.center, element_pos, np.arange(0, n_elem))
    else:
        assert False


def set_aperture_masks(sequence, probe) -> TxRxSequence:
    def get_new_ap_if_necessary(ap):
        if isinstance(ap, Aperture):
            center_element = __get_aperture_center_element(ap, probe)
            return __get_aperture_mask_with_padding(
                center_element=center_element,
                size=ap.size,
                probe_model=probe
            )
        else:
            return ap, (0, 0)

    new_ops = []
    for i, op in enumerate(sequence.ops):
        # Replace
        old_tx = op.tx
        old_rx = op.rx
        new_tx_ap, _ = get_new_ap_if_necessary(old_tx.aperture)
        new_rx_ap, padding = get_new_ap_if_necessary(old_rx.aperture)

        new_tx = dataclasses.replace(old_tx, aperture=new_tx_ap)
        new_rx = dataclasses.replace(old_rx, aperture=new_rx_ap,
                                     padding=padding)
        new_op = dataclasses.replace(op, tx=new_tx, rx=new_rx)
        new_ops.append(new_op)
    return dataclasses.replace(sequence, ops=new_ops)


def __get_aperture_mask_with_padding(center_element, size, probe_model):
    n_elem = probe_model.n_elements
    if size == 0:
        return np.zeros(size).astype(bool), (0, 0)
    if size is None:
        size = n_elem
    left_half_size = (size - 1) / 2  # e.g. size 32 -> 15, size 33 -> 16
    # left side:
    origin = int(round(center_element - left_half_size + 1e-9))  # e.g. center 0 -> origin -15
    actual_origin = max(0, origin)
    left_padding = abs(min(origin, 0))  # origin -15 -> left padding 15
    # right side
    # aperture last element, e.g. center 0, size 32 -> 16
    end = origin + size - 1
    actual_end = min(n_elem - 1, end)
    right_padding = abs(min(actual_end - end, 0))
    aperture = np.zeros((n_elem,), dtype=bool)
    aperture[int(actual_origin):(int(actual_end) + 1)] = True
    return aperture, (left_padding, right_padding)


def __get_sample_range(op: TxRx, tx_delay_center, fs):
    sample_range = op.rx.sample_range
    init_delay = op.rx.init_delay
    pulse = op.tx.excitation
    if init_delay == "tx_start":
        return sample_range
    elif init_delay == "tx_center":
        delay = get_init_delay(pulse, tx_delay_center)  # [s]
        delay = delay * fs / op.rx.downsampling_factor
        return tuple(int(round(v + delay)) for v in sample_range)
    else:
        raise ValueError(f"Unrecognized value '{init_delay}' for init_delay.")


def __get_sampling_frequency(context: KernelExecutionContext):
    if context.hardware_ddc is not None:
        return context.device.sampling_frequency / context.hardware_ddc.decimation_factor
    else:
        return context.device.sampling_frequency


def get_init_delay(pulse, tx_delay_center):
    """
    Returns burst factor [s]
    """
    n_periods = pulse.n_periods
    fc = pulse.center_frequency
    burst_factor = n_periods / (2 * fc)
    delay = burst_factor + tx_delay_center
    return delay


def get_center_delay(sequence: TxRxSequence, probe_model):
    """
    NOTE: the input sequence TX must be defined by focus, angle speed of sound
    This function does not work with the tx rx sequence with raw delays.
    """
    sequence_with_masks: TxRxSequence = set_aperture_masks(
        sequence=sequence,
        probe=probe_model
    )
    _, center_delay = get_tx_delays(probe_model, sequence, sequence_with_masks)
    return center_delay
