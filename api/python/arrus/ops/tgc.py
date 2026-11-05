import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class LinearTgc:
    """
    Set linear TGC on the device.

    :param start: tgc starting gain [dB]
    :param slope: tgc gain slope [dB/m]
    :param clip: whether the TGC values should be clipped to the range of gain values available on the device
    :param use_medium: whether the medium speed of sound should be used
    :param name: the name of the TGC curve; the name can be used to update the TGC curve after uploading
      the sequence (see Us4R.set_tgc); None means that no name is given (for a curve set for the whole
      TX/RX sequence this means "default")
    """
    start: float
    slope: float
    clip: bool = True
    use_medium: bool = True
    name: Optional[str] = None
