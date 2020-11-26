import dataclasses


@dataclasses.dataclass(frozen=True)
class LinearTgc:
    """
    Set linear TGC on the device.

    :param start: tgc starting gain [dB]
    :param slope: tgc gain slope [dB/m]
    """
    start: float
    slope: float