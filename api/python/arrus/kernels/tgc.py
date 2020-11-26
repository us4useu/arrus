import numpy as np


def compute_linear_tgc(seq_context, linear_tgc):
    seq = seq_context.op
    tgc_start = linear_tgc.start
    tgc_slope = linear_tgc.slope
    sample_range = seq.rx_sample_range
    start_sample, end_sample = sample_range
    downsampling_factor = seq.downsampling_factor
    fs = seq_context.device.sampling_frequency/seq.downsampling_factor
    # medium parameters
    c = seq.speed_of_sound
    if c is None:
        c = seq_context.medium.speed_of_sound

    distance = np.arange(start=round(400/downsampling_factor),
                         stop=end_sample,
                         step=round(150/downsampling_factor))/fs*c

    return tgc_start + distance*tgc_slope
