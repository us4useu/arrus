import numpy as np


def compute_linear_tgc(sample_range, fs, downsampling_factor, speed_of_sound, linear_tgc):
    tgc_start = linear_tgc.start
    tgc_slope = linear_tgc.slope
    start_sample, end_sample = sample_range
    # medium parameters
    c = speed_of_sound

    distance = np.arange(start=round(400/downsampling_factor),
                         stop=end_sample,
                         step=round(153/downsampling_factor))/fs*c

    return tgc_start + distance*tgc_slope
