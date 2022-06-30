import numpy as np


def compute_linear_tgc(seq_context, fs, linear_tgc):
    tgc_start = linear_tgc.tgc_start
    tgc_slope = linear_tgc.tgc_slope

    if tgc_start is None or tgc_slope is None:
        return [], []
    seq = seq_context.op
    sample_range = seq.rx_sample_range
    start_sample, end_sample = sample_range
    # medium parameters
    c = seq.speed_of_sound
    if c is None:
        c = seq_context.medium.speed_of_sound

    sampling_time = np.arange(0, end_sample, 50)/fs  # Arbitrary sampling freq.
    distance = sampling_time*c
    return sampling_time, tgc_start + distance*tgc_slope
