import numpy as np


def compute_linear_tgc(seq_context, fs, linear_tgc):
    tgc_start = linear_tgc.start
    tgc_slope = linear_tgc.slope

    if tgc_start is None or tgc_slope is None:
        return [], []
    seq = seq_context.op
    sample_range = seq.rx_sample_range
    start_sample, end_sample = sample_range
    # medium parameters
    c = seq.speed_of_sound
    if c is None:
        c = seq_context.medium.speed_of_sound
    ds = 64  # Arbitrary TGC curve sampling
    sampling_time = np.arange(0, end_sample, ds)
    sampling_time = np.append(sampling_time, [end_sample-1])
    sampling_time = sampling_time/fs
    distance = sampling_time*c
    return sampling_time, tgc_start + distance*tgc_slope
