import numpy as np


def compute_linear_tgc(seq_context):
    op = seq_context.op
    tgc_start = op.tgc_start
    tgc_slope = op.tgc_slope
    sample_range = op.rx_sample_range
    start_sample, end_sample = sample_range
    downsampling_factor = op.downsampling_factor
    fs = seq_context.device.sampling_frequency/op.downsampling_factor
    # medium parameters
    c = op.speed_of_sound
    if c is None:
        c = seq_context.medium.speed_of_sound

    distance = np.arange(start=round(400/downsampling_factor),
                         stop=end_sample,
                         step=round(150/downsampling_factor))/fs*c

    return tgc_start + distance*tgc_slope
