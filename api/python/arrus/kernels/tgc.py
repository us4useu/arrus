import numpy as np
import arrus.ops.imaging
import arrus.ops.us4r


def compute_linear_tgc(seq_context, linear_tgc):
    seq = seq_context.op

    if isinstance(seq, arrus.ops.imaging.SimpleTxRxSequence):
        sample_range = seq.rx_sample_range
        start_sample, end_sample = sample_range
        downsampling_factor = seq.downsampling_factor
        fs = seq_context.device.sampling_frequency/downsampling_factor
        c = seq.speed_of_sound
        if c is None:
            c = seq_context.medium.speed_of_sound
    elif isinstance(seq, arrus.ops.us4r.TxRxSequence):
        if len(seq.ops) == 0:
            raise ValueError("The sequence should have at least one op.")
        reference_rx = seq.ops[0].rx
        sample_range = reference_rx.sample_range
        start_sample, end_sample = sample_range
        downsampling_factor = reference_rx.downsampling_factor
        fs = seq_context.device.sampling_frequency/downsampling_factor
        if seq_context.medium is None:
            raise ValueError("Medium must be specified in order to set LinearTGC "
                             "for custom TxRxSequence.")
        else:
            c = seq_context.medium.speed_of_sound
    else:
        raise ValueError(f"Unsupported sequence type: {type(seq)}")

    tgc_start = linear_tgc.start
    tgc_slope = linear_tgc.slope

    distance = np.arange(start=round(400/downsampling_factor),
                         stop=end_sample,
                         step=round(150/downsampling_factor))/fs*c

    return tgc_start + distance*tgc_slope
