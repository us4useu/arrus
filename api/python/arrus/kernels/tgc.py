from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TgcCalculationContext:
    end_sample: int
    speed_of_sound: float


def compute_linear_tgc(
        tgc_context: TgcCalculationContext,
        fs: float,
        linear_tgc,
):
    tgc_start = linear_tgc.start
    tgc_slope = linear_tgc.slope
    if tgc_start is None or tgc_slope is None:
        return [], []
    end_sample = tgc_context.end_sample
    c = tgc_context.speed_of_sound
    ds = 64  # Arbitrary linear TGC curve sampling
    sampling_time = np.arange(0, end_sample, ds)
    sampling_time = np.append(sampling_time, [end_sample-1])
    sampling_time = sampling_time/fs
    distance = sampling_time*c
    tgc_values = tgc_start + distance*tgc_slope
    return sampling_time, tgc_values


def _get_rx_end_sample(rx, fs, speed_of_sound):
    from arrus.kernels.tx_rx_sequence import (
        convert_depth_to_sample_range, convert_time_to_sample_range)
    if rx.sample_range is not None:
        return rx.sample_range[1]
    elif rx.depth_range is not None:
        if speed_of_sound is None:
            raise ValueError("Speed of sound is required to convert RX depth range to samples.")
        return convert_depth_to_sample_range(rx.depth_range, fs=fs, speed_of_sound=speed_of_sound)[1]
    elif rx.time_range is not None:
        return convert_time_to_sample_range(rx.time_range, fs=fs)[1]
    else:
        raise ValueError("RX sample range is not specified.")


def resolve_tgc(tgc, end_sample: int, fs: float, speed_of_sound, gain_range=None):
    """
    Converts the given TGC description to arrus.ops.us4r.Tgc.

    :param tgc: None, arrus.ops.us4r.Tgc, arrus.ops.tgc.LinearTgc or a list of values
    :param end_sample: the end sample of the RX (the TGC curve will be defined up to this sample)
    :param fs: the sampling frequency of the RX data [Hz]
    :param speed_of_sound: speed of sound [m/s] (required for LinearTgc)
    :param gain_range: a pair (min, max) gain [dB] available on the device; if provided, LinearTgc curves with
      clip=True will be clipped to this range
    :return: arrus.ops.us4r.Tgc or None, if no TGC curve was specified
    """
    import arrus.ops.us4r
    import arrus.ops.tgc
    if tgc is None:
        return None
    if isinstance(tgc, arrus.ops.us4r.Tgc):
        return tgc
    if isinstance(tgc, arrus.ops.tgc.LinearTgc):
        if tgc.start is None or tgc.slope is None:
            return None
        if speed_of_sound is None:
            raise ValueError("Speed of sound is required to compute linear TGC "
                             "(please specify the medium).")
        t, values = compute_linear_tgc(
            TgcCalculationContext(end_sample=end_sample, speed_of_sound=speed_of_sound),
            fs, tgc)
        values = np.asarray(values)
        if tgc.clip and gain_range is not None:
            values = np.clip(values, gain_range[0], gain_range[1])
        return arrus.ops.us4r.Tgc(values=values, time=t, name=tgc.name)
    values = np.asarray(tgc)
    if values.size == 0:
        return None
    return arrus.ops.us4r.Tgc(values=values)


def resolve_sequence_tgcs(sequence, fs_for_rx, speed_of_sound, gain_range=None):
    """
    Resolves all the TGC curves of the given arrus.ops.us4r.TxRxSequence (sequence-level and Rx-level)
    to arrus.ops.us4r.Tgc objects (LinearTgc -> Tgc with time points).

    :param fs_for_rx: function rx -> sampling frequency of the RX data
    :return: a new TxRxSequence
    """
    import dataclasses
    seq_tgc = sequence.get_tgc()
    if seq_tgc is not None:
        rx = sequence.ops[0].rx
        fs = fs_for_rx(rx)
        seq_tgc = resolve_tgc(seq_tgc, _get_rx_end_sample(rx, fs, speed_of_sound), fs, speed_of_sound, gain_range)
    new_ops = []
    for op in sequence.ops:
        rx = op.rx
        if rx.tgc is not None:
            fs = fs_for_rx(rx)
            rx_tgc = resolve_tgc(rx.tgc, _get_rx_end_sample(rx, fs, speed_of_sound), fs, speed_of_sound, gain_range)
            rx = dataclasses.replace(rx, tgc=rx_tgc)
            op = dataclasses.replace(op, rx=rx)
        new_ops.append(op)
    return dataclasses.replace(sequence, ops=new_ops, tgc_curve=seq_tgc if seq_tgc is not None else [])
