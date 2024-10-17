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
from arrus.kernels.kernel import KernelExecutionContext, ConversionResults
from arrus.framework import Constant
import math


def _sort_txs_by_aperture(sequence, probe_tx):
    def get_aperture_left(aperture):
        mask, _, = get_new_masked_aperture_if_necessary(aperture, probe_tx)
        return np.min(np.argwhere(mask.flatten()))

    new_ops = []
    for op in sequence.ops:
        tx = op.tx
        if isinstance(tx, Tx):
            tx = [tx]
        txs = list(tx)
        txs = [(tx, get_aperture_left(tx.aperture)) for tx in txs]
        txs = sorted(txs, key=lambda x: x[1])
        txs = list(zip(*txs))[0]
        op = dataclasses.replace(op, tx=txs)
        new_ops.append(op)
    sequence = dataclasses.replace(sequence, ops=new_ops)
    return sequence


def process_tx_rx_sequence(context: KernelExecutionContext):
    """
    Performs all the necessary pre-processing needed to
    run raw TX/RX sequence on the us4r device.

    Currently, this function only translates (focus, angle, speed_of_sound)
    into a list of raw delays.
    """
    sequence: TxRxSequence = context.op
    tx_delay_constants = context.constants
    c = context.medium.speed_of_sound if context.medium is not None else None

    # Determine unique probe tx and rx id.
    probe_tx_id = sequence.get_tx_probe_id_unique()
    probe_rx_id = sequence.get_rx_probe_id_unique()
    probe_tx = context.device.get_probe_by_id(probe_tx_id).model
    probe_rx = context.device.get_probe_by_id(probe_rx_id).model
    fs: float = __get_sampling_frequency(context)
    # Sort Txs of each op by the first active element
    sequence = _sort_txs_by_aperture(sequence, probe_tx)
    # Update the following sequence parameters (if necessary):
    # - tx: aperture (to binary mask)
    # - rx: aperture (to binary mask), rx padding
    sequence, _, constants = convert_to_us4r_sequence_with_constants(
        sequence=sequence,
        probe_tx=probe_tx,
        probe_rx=probe_rx,
        fs=fs,
        tx_focus_constants=tx_delay_constants,
        speed_of_sound=c
    )
    return ConversionResults(
        sequence=sequence,
        constants=constants
    )


def convert_to_us4r_sequence(sequence: TxRxSequence, probe_tx, probe_rx, fs: float):
    """
    for backward compatibility
    """
    seq, center_delay, constants = convert_to_us4r_sequence_with_constants(
        sequence=sequence,
        probe_tx=probe_tx,
        probe_rx=probe_rx,
        fs=fs,
        tx_focus_constants=(),
        speed_of_sound=None
    )
    return seq, center_delay


def _get_full_tx_delays(constant_delays, sequence_with_masks):
    """
    :return: array (n ops, n elements)
    """
    output_array = []
    for i, op in enumerate(sequence_with_masks.ops):
        tx = op.tx
        core_delays = np.zeros(tx.aperture.shape, dtype=np.float32)
        core_delays[tx.aperture] = constant_delays[i]
        output_array.append(core_delays)
    return np.stack(output_array)


def __merge_txs(txs):
    """
    NOTE: assuming TXs are already sorted by the position of the first active element!
    """
    if len(txs) == 1:
        return txs[0]

    # Validate
    __assert_apertures_disjoint([tx.aperture for tx in txs])
    # assert sorted by the position of the first active element
    is_all_sorted_by_first_element = (np.diff([np.min(np.argwhere(tx.aperture)) for tx in txs]) > 0).all()
    assert is_all_sorted_by_first_element, "Internal error: txs: were not sorted appropriately"
    # assert unique values: excitation, placement
    n_unqiue_pulses = len({tx.excitation for tx in txs})  # NOTE: assuming tx.pulse is hashable
    if n_unqiue_pulses > 1:
        raise ValueError("Each TX of a single op should have exactly the same Pulse definition")
    n_unique_placements = len({tx.placement for tx in txs})
    if n_unique_placements > 1:
        raise ValueError("Each Tx of a single op should have exactly the same placement.")

    ref_tx = txs[0]
    aperture = ref_tx.aperture
    delays = np.zeros(len(aperture.flatten()), dtype=np.float32)
    delays[:] = np.nan
    for tx in txs:
        aperture = np.logical_or(aperture, tx.aperture)
        for d, i in zip(tx.delays, np.argwhere(tx.aperture).flatten().tolist()):
            delays[i] = d
    # Remove unused elements
    delays = delays[np.logical_not(np.isnan(delays))]
    return dataclasses.replace(ref_tx, aperture=aperture, delays=delays)


def convert_to_us4r_sequence_with_constants(
        sequence: TxRxSequence, probe_tx, probe_rx, fs: float,
        tx_focus_constants, speed_of_sound
):
    """
    NOTE: assumptions regarding sequence txs:
    - each tx is actually a list of txs (even a single-element tx)
    - txs are sorted by the position of the first element in the aperture
    """
    sequence_with_masks: TxRxSequence = set_aperture_masks(
        sequence=sequence,
        probe_tx=probe_tx,
        probe_rx=probe_rx
    )
    original_sequence = sequence
    dels, tx_center_delay = _get_tx_delays_internal(
        probe=probe_tx,
        sequence=sequence,
        seq_with_masks=sequence_with_masks,
    )
    # Calculate delays for each constant.
    new_ops = []
    # Update input sequence.
    for i, op in enumerate(sequence_with_masks.ops):
        # Replace the input txs with the txs with the raw tx delays
        new_txs = []
        for j, tx in enumerate(op.tx):
            d = dels[i][j]
            new_tx = dataclasses.replace(tx, delays=np.atleast_1d(d),
                                         focus=None, angle=None,
                                         speed_of_sound=None)
            new_txs.append(new_tx)
        # NOTE: assuming that new_txs are sorted by the first element of aperture!
        new_tx = __merge_txs(new_txs)

        sample_range = __get_sample_range(
            rx=op.rx, tx=new_tx, tx_delay_center=tx_center_delay, fs=fs,
            c=speed_of_sound
        )

        old_rx = op.rx
        new_rx = dataclasses.replace(old_rx,
                                     sample_range=sample_range,
                                     depth_range=None,
                                     init_delay="tx_start")
        new_op = dataclasses.replace(op, tx=new_tx, rx=new_rx)
        new_ops.append(new_op)

    sequence = dataclasses.replace(sequence, ops=new_ops)

    output_constants = []
    for i, tx_focus_const in enumerate(tx_focus_constants):
        focus = tx_focus_const.value
        focuses = [focus]*len(sequence_with_masks.ops)
        constant_delays, _ = get_tx_delays_for_focuses(
            probe=probe_tx,
            sequence=original_sequence,
            seq_with_masks=sequence_with_masks,
            tx_focuses=focuses
        )
        full_tx_delays = _get_full_tx_delays(
            constant_delays, sequence_with_masks)
        output_constants.append(
            Constant(
                value=full_tx_delays,
                placement=tx_focus_const.placement,
                name=f"sequence/txDelays:{i}"
            )
        )
    return sequence, tx_center_delay, output_constants


def _get_tx_delays_internal(
        probe, sequence: TxRxSequence, seq_with_masks: TxRxSequence,
):
    focuses = [[tx.focus for tx in op.tx] if isinstance(op.tx, typing.Iterable) else [op.tx.focus]
               for op in sequence.ops]
    return get_tx_delays_for_focuses(
        probe, sequence, seq_with_masks, focuses)


def get_tx_delays(
        probe, sequence: TxRxSequence, seq_with_masks: TxRxSequence,
):
    sequence = _sort_txs_by_aperture(sequence, probe)
    seq_with_masks = _sort_txs_by_aperture(seq_with_masks, probe)
    return _get_tx_delays_internal(probe, sequence, seq_with_masks)


def get_tx_delays_for_focuses(
        probe, sequence: TxRxSequence, seq_with_masks: TxRxSequence,
        tx_focuses
):
    """
    Returns tx_center_delay = None when all TXs have empty aperture.

    For ops with multiple TX: the TX delays will be equalized to the MAXIMUM CENTER DELAY of the whole op.
    """
    # COMPUTE TX APERTURE CENTERS.
    tx_aperture_center_element = []
    for op in sequence.ops:
        op_centers = []
        for tx in op.tx:
            ap_cent_elem = __get_center_element(
                tx.aperture, probe_model=probe
            )
            op_centers.append(ap_cent_elem)
        tx_aperture_center_element.append(np.asarray(op_centers))

    # COMPUTE TX DELAYS.
    tx_center_angles, tx_center_x, tx_center_z = \
        get_aperture_center(tx_aperture_center_element, probe)
    tx_delays = []  # (ntx,), each element: list of (n_elem delays). NOTE: n_elem is the number of probe elements
    center_delays = []  # (ntx, )
    element_x, element_z = probe.element_pos_x, probe.element_pos_z
    element_x, element_z = np.atleast_2d(element_x), np.atleast_2d(element_z)  # [1, n_elem]
    # Compute tx_delays
    for i, op in enumerate(sequence.ops):
        op_delays = []
        op_center_delays = []
        for j, tx in enumerate(op.tx):
            if tx.delays is not None:
                # RAW DELAYS
                delays = tx.delays
                center_delay = None
            else:
                tx_angle = tx.angle + tx_center_angles[i][j]
                tx_cent_x = tx_center_x[i][j]
                tx_cent_z = tx_center_z[i][j]
                c = tx.speed_of_sound
                tx_focus = tx_focuses[i][j]
                if np.isinf(tx_focus):
                    # PWI
                    delays = (element_x * np.sin(tx_angle)
                              + element_z * np.cos(tx_angle)) / c  # [1, n_elem]
                    center_delay = (tx_cent_x * np.sin(tx_angle)
                                    + tx_cent_z * np.cos(tx_angle)) / c  # scalar
                else:
                    # Virtual source/focus
                    focus_x = tx_cent_x + tx_focus * np.sin(tx_angle)  # scalar
                    focus_z = tx_cent_z + tx_focus * np.cos(tx_angle)  # scalar
                    delays = np.sqrt((focus_x - element_x) ** 2
                                     + (focus_z - element_z) ** 2) / c  # [1, n_elem]
                    center_delay = np.sqrt((focus_x - tx_cent_x) ** 2
                                           + (focus_z - tx_cent_z) ** 2) / c  # scalar
                    foc_defoc = 1 - 2*float(tx_focus > 0)
                    delays = delays * foc_defoc
                    center_delay = center_delay * foc_defoc

            op_delays.append(np.squeeze(delays))
            op_center_delays.append(center_delay)
        tx_delays.append(op_delays)
        # EQUALIZE TO THE MAXIMUM CENTER DELAY OF THE WHOLE OPERATION
        center_delays.append(op_center_delays)

    normalized_delays = []
    normalized_center_delays = []

    # Equalize delays to the TX center.
    for i, op in enumerate(sequence.ops):
        op_normalized_delays = []
        op_normalized_center_delays = []
        for j, tx in enumerate(op.tx):
            op_center_delay = center_delays[i][j]
            op_delays = tx_delays[i][j]
            tx_aperture_mask = seq_with_masks.ops[i].tx[j].aperture
            if op_center_delay is None:
                # RAW DELAYS pass-through. We do not equalize them.
                op_normalized_delays.append(op_delays)
                op_normalized_center_delays.append(np.nan)  # Do not equalize through the whole sequence
            else:
                # Delays calculated based on the focus,angle, sos.
                # Use only delays for the active elements.
                op_delays = op_delays[tx_aperture_mask]
                if len(op_delays) == 0:
                    # Empty TX aperture.
                    op_normalized_delays.append(op_delays)
                    op_normalized_center_delays.append(np.nan)
                else:
                    # Move tx delays to bias = 0.
                    delays_min = np.min(op_delays).item()
                    op_delays = op_delays - delays_min
                    op_center_delay = op_center_delay - delays_min
                    op_normalized_delays.append(op_delays)
                    op_normalized_center_delays.append(op_center_delay)
        normalized_delays.append(op_normalized_delays)
        normalized_center_delays.append(op_normalized_center_delays)

    # NOTE: HERE THE NORMALIZED_CENTER_DELAYS BECOME A LIST OF FLOAT VALUES (N_TX, )
    # Equalize TX delays for each op to have the same delay in the center of the aperture
    for i, op in enumerate(sequence.ops):
        if np.isnan(normalized_center_delays[i]).all():
            # Nothing to equalize, pass-through delays
            normalized_center_delays[i] = np.nan
        else:
            max_center_delay = np.nanmax(normalized_center_delays[i])
            for j, tx in enumerate(op.tx):
                if not np.isnan(normalized_center_delays[i][j]):
                    d = normalized_delays[i][j]
                    # Move back to the zero in the center of the aperture
                    d = d - normalized_center_delays[i][j]
                    # Move center delay to the maximum value
                    d = d + max_center_delay
                    normalized_delays[i][j] = d

            normalized_center_delays[i] = max_center_delay

    # Equalize through the whole sequence (NOTE: RX active operations only!)
    # The common delay applied for center of each TX aperture
    # So we can use a single TX center delay when RX beamforming the data.
    # The center of transmit will be in the same position for all TX/RXs.
    # Note: in the case when all TXs have empty TX aperture, None should be
    # returned.
    is_nothing_to_equalize = np.asarray([
        np.isnan(d) for d in normalized_center_delays
    ]).all()

    non_empty_rx_ops = [i for i, op in enumerate(seq_with_masks.ops) if np.sum(op.rx.aperture) > 0]

    if is_nothing_to_equalize:
        tx_center_delay = None
    else:
        tx_center_delay = np.nanmax(np.asarray(normalized_center_delays)[non_empty_rx_ops])
    equalized_tx_delays = []

    for i, (op, op_with_mask) in enumerate(zip(sequence.ops, seq_with_masks.ops)):
        op_equalized_tx_delays = []
        for j, tx in enumerate(op.tx):
            d = normalized_delays[i][j]
            cd = normalized_center_delays[i]
            if not np.isnan(cd) and len(d) > 0 and np.sum(op_with_mask.rx.aperture) > 0:
                # Non-empty delays.
                d = d - normalized_center_delays[i] + tx_center_delay
            op_equalized_tx_delays.append(d)
        equalized_tx_delays.append(op_equalized_tx_delays)

    return equalized_tx_delays, tx_center_delay


def get_aperture_center(tx_aperture_center_element, probe):
    """
    Interpolates given TX aperture center elements into the positions
    in a probe's coordinate system.
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

    tx_aperture_center_angle, tx_aperture_center_x, tx_aperture_center_z = [], [], []
    for center_elements in tx_aperture_center_element:
        ap_angle = np.interp(center_elements,
                             np.arange(0, n_elements), angle)
        ap_center_z = np.interp(center_elements,
                             np.arange(0, n_elements),
                             np.squeeze(probe.element_pos_z))
        ap_center_x = np.interp(center_elements,
                             np.arange(0, n_elements),
                             np.squeeze(probe.element_pos_x))
        tx_aperture_center_angle.append(ap_angle)
        tx_aperture_center_z.append(ap_center_z)
        tx_aperture_center_x.append(ap_center_x)
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


def __get_tx_list(tx):
    if isinstance(tx, Tx):
        return [tx]
    elif isinstance(tx, typing.Iterable):
        return tx
    else:
        raise ValueError(f"Invalid type of TX: {tx}")


def __assert_apertures_disjoint(apertures):
    apertures = np.stack(apertures).astype(np.int32)
    ntimes = np.sum(apertures, axis=0)
    if not (ntimes <= 1).all():
        raise ValueError(f"All TX/RX aperture should be disjoint, "
                         f"got: {apertures}")


def get_new_masked_aperture_if_necessary(ap, probe):
    if isinstance(ap, Aperture):
        center_element = __get_aperture_center_element(ap, probe)
        return __get_aperture_mask_with_padding(
            center_element=center_element,
            size=ap.size,
            probe_model=probe
        )
    else:
        return ap, (0, 0)


def set_aperture_masks(sequence, probe_tx, probe_rx) -> TxRxSequence:
    new_ops = []
    for i, op in enumerate(sequence.ops):
        # Replace
        old_tx = __get_tx_list(op.tx)
        new_txs = []

        for tx in old_tx:
            new_tx_ap, _ = get_new_masked_aperture_if_necessary(tx.aperture, probe=probe_tx)
            new_tx = dataclasses.replace(tx, aperture=new_tx_ap)
            new_txs.append(new_tx)

        old_rx = op.rx
        new_rx_ap, padding = get_new_masked_aperture_if_necessary(old_rx.aperture,
                                                                  probe=probe_rx)

        new_rx = dataclasses.replace(old_rx, aperture=new_rx_ap,
                                     padding=padding)
        new_op = dataclasses.replace(op, tx=new_txs, rx=new_rx)
        new_ops.append(new_op)
    return dataclasses.replace(sequence, ops=new_ops)


def __get_aperture_mask_with_padding(center_element, size, probe_model):
    n_elem = probe_model.n_elements
    if size == 0:
        return np.zeros(n_elem).astype(bool), (0, 0)
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


def __get_sample_range(rx, tx, tx_delay_center, fs, c):
    sample_range = rx.sample_range
    if rx.depth_range is not None:
        sample_range = convert_depth_to_sample_range(rx.depth_range, fs=fs,
                                                     speed_of_sound=c)
    init_delay = rx.init_delay
    pulse = tx.excitation
    if init_delay == "tx_start":
        return sample_range
    elif init_delay == "tx_center":
        delay = get_init_delay(pulse, tx_delay_center)  # [s]
        delay = delay * fs / rx.downsampling_factor
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


def get_center_delay(sequence: TxRxSequence, probe_tx, probe_rx):
    """
    NOTE: the input sequence TX must be defined by focus, angle speed of sound
    This function does not work with the tx rx sequence with raw delays.
    """
    sequence = _sort_txs_by_aperture(sequence, probe_tx)
    sequence_with_masks: TxRxSequence = set_aperture_masks(
        sequence=sequence,
        probe_tx=probe_tx,
        probe_rx=probe_rx
    )
    _, center_delay = _get_tx_delays_internal(probe_tx, sequence, sequence_with_masks)
    return center_delay


def convert_depth_to_sample_range(depth_range, fs, speed_of_sound):
    """
    Converts depth range (in [m]) to the sample range
    (in the number of samples).
    """
    sample_range = np.round(2*fs*np.asarray(depth_range)/speed_of_sound).astype(int)
    # Round the number of samples to a value divisible by 64.
    # Number of acquired must be divisible by 64 (required by us4R driver).
    n_samples = sample_range[1]-sample_range[0]
    n_samples = 64*int(math.ceil(n_samples/64))
    sample_range = sample_range[0], sample_range[0]+n_samples
    return sample_range


def get_tx_rx_sequence_sample_range(seq: TxRxSequence, fs, speed_of_sound):
    """
    Returns sample range (if provided) or returns depth range converted
    to the sample range according to the given sampling frequency
    speed of sound.
    """
    op = seq.ops[0].rx
    if op.sample_range is not None:
        return op.sample_range
    else:
        return convert_depth_to_sample_range(
            depth_range=op.depth_range,
            fs=fs,
            speed_of_sound=speed_of_sound
        )
