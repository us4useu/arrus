from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TgcCalculationContext:
    end_sample: int
    speed_of_sound: float


def compute_linear_tgc(
        tgc_context: TgcCalculationContext,
        fs: float,
        linear_tgc
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
    # TODO: the below should be moved to ARRUS CORE
    if linear_tgc.clip:
        tgc_values = np.clip(tgc_values, 14, 54)
    return sampling_time, tgc_values
