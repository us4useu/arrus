from dataclasses import dataclass
from collections.abc import Iterable
import abc
import numpy as np


class Excitation(abc.ABC):
    """
    An excitation (a signal) to perform.
    """
    pass


@dataclass(frozen=True)
class SineWave(Excitation):
    """
    Sine wave excitation.

    :var frequency: transmitted carrier/nominal/center frequency [Hz]
    :var n_periods: number of sine periods in the transmitted burst, can be fractional
    :var inverse: whether the resulting wave should be inverted
    """
    frequency: float
    n_periods: float
    inverse: bool


class Aperture(abc.ABC):
    """
    An aperture - set of channels to perform.

    This class is abstract and should not be instantiated.
    """
    pass


@dataclass(frozen=True)
class RegionBasedAperture(Aperture):
    """
    A region-based aperture.

    The aperture represents a single, contiguous range of channels.
    :var origin: an origin channel of the aperture
    :var size: a length of the aperture
    """
    origin: int
    size: int


@dataclass(frozen=True)
class MaskAperture(Aperture):
    """
    A mask-based aperture.

    This aperture can represent a (possibly non-contiguous) set of channels,
    for example:

    ::
        [1,1,0,0,1,1]

    represents channels 0, 1, 4, 5

    :var mask: a mask to set, one dimensional numpy array
    """
    mask: np.ndarray


@dataclass(frozen=True)
class SingleElementAperture(Aperture):
    """
    Represents an aperture consisting of a single element.

    :var mask: an element of the aperture
    """
    element: int

