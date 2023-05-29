import numpy as np
import arrus.ops.imaging
import arrus.ops.us4r


def compute_linear_tgc(seq_context, fs, linear_tgc):
    tgc_start = linear_tgc.start
    tgc_slope = linear_tgc.slope

    if tgc_start is None or tgc_slope is None:
        return [], []
    seq = seq_context.op
    if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
        sample_range = seq.rx_sample_range
        c = seq.speed_of_sound
        if c is None:
            c = seq_context.medium.speed_of_sound
    elif isinstance(seq, arrus.ops.us4r.TxRxSequence):
        sample_range = seq.get_sample_range_unique()
        if seq_context.medium is None:
            raise ValueError(
                "Medium definition is required for custom tx/rx sequence "
                "when setting linear TGC.")
        c = seq_context.medium.speed_of_sound
    else:
        raise ValueError(f"Unsupported type of TX/RX sequence: {type(seq)}")
    start_sample, end_sample = sample_range
    ds = 64  # Arbitrary TGC curve sampling
    sampling_time = np.arange(0, end_sample, ds)
    sampling_time = np.append(sampling_time, [end_sample-1])
    sampling_time = sampling_time/fs
    distance = sampling_time*c
    tgc_values = tgc_start + distance*tgc_slope
    # TODO: the below should be moved to ARRUS CORE
    if linear_tgc.clip:
        tgc_values = np.clip(tgc_values, 14, 54)
    return sampling_time, tgc_values
