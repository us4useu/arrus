import dataclasses
import math
from collections.abc import Collection
import numpy as np
import arrus.exceptions
from arrus.ops.us4r import (
    Tx, Rx, TxRx, TxRxSequence, Pulse, Aperture
)
from arrus.ops.imaging import (
    LinSequence, PwiSequence, StaSequence, SimpleTxRxSequence
)
from arrus.kernels.kernel import KernelExecutionContext, ConversionResults
from arrus.kernels.tx_rx_sequence import (
    process_tx_rx_sequence,
    get_tx_delays,
    set_aperture_masks
)
from arrus.framework.constant import Constant, _get_unique_name


def _get_sampling_frequency(context: KernelExecutionContext):
    if context.hardware_ddc is not None:
        decimation_factor = context.hardware_ddc.decimation_factor
    else:
        seq: SimpleTxRxSequence = context.op
        decimation_factor = seq.downsampling_factor
    return context.device.sampling_frequency / decimation_factor


def process_simple_tx_rx_sequence(context: KernelExecutionContext):
    """
    Converts SimpleTxRxSequence to raw TxRxSequence.

    :param context: execution context
    """
    op = context.op
    if not isinstance(op, SimpleTxRxSequence):
        raise ValueError("The provided op should be "
                         "an instance of SimpleTxRxSequence (or derived) class.")

    # Medium
    c = __get_speed_of_sound(context)
    if op.tx_placement != op.rx_placement:
        raise ValueError("TX and RX should be done on the same Probe.")
    probe = context.device.get_probe_by_id(op.tx_placement).model
    # TX/RX
    new_seq = convert_to_tx_rx_sequence(
        c, op, probe, fs=_get_sampling_frequency(context))
    new_context = dataclasses.replace(context, op=new_seq)
    # Process the raw sequence, to calculate all the necessary
    # TX delays.
    return process_tx_rx_sequence(new_context)


def get_center_delay(sequence: SimpleTxRxSequence, c: float, probe_model, fs):
    # NOTE: this method will work only properly only in the case of a single TX focus.
    tx_rx_sequence = convert_to_tx_rx_sequence(
        c, sequence, probe_model, fs=fs)
    sequence_with_masks: TxRxSequence = set_aperture_masks(
        sequence=tx_rx_sequence,
        probe_tx=probe_model,
        probe_rx=probe_model
    )
    _, center_delay = get_tx_delays(probe_model, tx_rx_sequence,
                                    sequence_with_masks)
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


def get_sample_range(op: SimpleTxRxSequence, fs, speed_of_sound):
    """
    Returns sample range (if provided) or returns depth range converted
    to the sample range according to the given sampling frequency
    speed of sound.
    """
    if op.rx_sample_range is not None:
        return op.rx_sample_range
    else:
        return convert_depth_to_sample_range(
            depth_range=op.rx_depth_range,
            fs=fs,
            speed_of_sound=speed_of_sound
        )


def convert_to_tx_rx_sequence(c: float, op: SimpleTxRxSequence, probe_model,
                              fs: float):
    tx_rx_params = compute_tx_rx_params(probe=probe_model, sequence=op)
    n_tx = len(tx_rx_params["tx_ap_cent"])
    txrx = []
    sample_range = get_sample_range(op, fs=fs, speed_of_sound=c)

    tx_focus = op.tx_focus
    if isinstance(tx_focus, arrus.framework.Constant):
        tx_focus = tx_focus.value

    for i in range(n_tx):
        tx_aperture = tx_rx_params["tx_apertures"][i]
        rx_aperture = tx_rx_params["rx_apertures"][i]
        tx_angle = tx_rx_params["tx_angle"][i]

        tx = Tx(tx_aperture, op.pulse,
                focus=tx_focus,
                angle=tx_angle,
                speed_of_sound=c,
                placement=op.tx_placement
        )
        rx = Rx(rx_aperture, sample_range, op.downsampling_factor,
                init_delay=op.init_delay, placement=op.rx_placement)
        txrx.append(TxRx(tx, rx, op.pri))
    # TGC curve should be set on later stage
    return TxRxSequence(txrx, tgc_curve=[], sri=op.sri, n_repeats=op.n_repeats)


def __convert_to_op_specific_constant(constant, op_i):
    component, variable = constant.name.split("/")
    output_name = f"{component}/op:{op_i}/{variable}"

    return arrus.framework.Constant(
        value=constant.value,
        placement=constant.placement,
        name=output_name
    )


def __get_speed_of_sound(context):
    if context.op.speed_of_sound is not None:
        return context.op.speed_of_sound
    else:
        return context.medium.speed_of_sound


def compute_tx_rx_params(probe, sequence: SimpleTxRxSequence):
    """
    Computes tx rx delays for provided angle and focus.
    """
    # INPUT
    tx_rxs = preprocess_sequence_parameters(probe, sequence)
    tx_angle = tx_rxs["tx_angle"]
    tx_focus = sequence.tx_focus
    tx_ap_cent = tx_rxs["tx_ap_cent"]  # center element
    tx_ap_size = tx_rxs["tx_ap_size"]
    rx_ap_cent = tx_rxs["rx_ap_cent"]  # center element
    rx_ap_size = tx_rxs["rx_ap_size"]
    # OUTPUT
    tx_apertures = []
    rx_apertures = []
    # APERTURES
    for tx_center, tx_size, rx_center, rx_size \
            in zip(tx_ap_cent, tx_ap_size, rx_ap_cent, rx_ap_size):
        tx_aperture = Aperture(center_element=tx_center, size=tx_size)
        rx_aperture = Aperture(center_element=rx_center, size=rx_size)
        tx_apertures.append(tx_aperture)
        rx_apertures.append(rx_aperture)

    tx_rxs["tx_apertures"] = tx_apertures
    tx_rxs["rx_apertures"] = rx_apertures
    return tx_rxs


def preprocess_sequence_parameters(probe_model, sequence: SimpleTxRxSequence):
    # Get default values and element position for the given probe.
    n_elem = probe_model.n_elements
    default_ap_cent = n_elem // 2 - 1
    default_ap_size = n_elem
    element_pos = (np.arange(0, n_elem) - (n_elem - 1) / 2) * probe_model.pitch

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
        "rx_ap_size": with_default(sequence.rx_aperture_size, default_ap_size)
    }
    # Broadcast the above values to the numpy.ndarray vectors,
    # all with the same length.
    # 1. Determine ndarray size.
    sizes = set([len(v) if isinstance(v, Collection) else 1
                 for _, v in tx_rxs.items()])
    if len(sizes) > 2 or (len(sizes) == 2 and 1 not in sizes):
        raise ValueError("All TX/RX parameters should be lists of the same "
                         f"sizes or scalars (found sizes: {sizes})")
    if 1 in sizes and len(sizes) == 2:
        sizes.remove(1)
    dst_size = next(iter(sizes))

    # 2. Do the broadcasting
    for k, v in tx_rxs.items():
        if not isinstance(v, np.ndarray):
            v = np.squeeze(np.array(v))
            v = np.atleast_1d(v)
        if len(v) == 1:
            v = np.repeat(v, dst_size)
        tx_rxs[k] = v
    return tx_rxs
