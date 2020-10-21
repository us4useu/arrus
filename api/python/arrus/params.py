from dataclasses import dataclass
from collections.abc import Iterable
import abc
import numpy as np


class Pulse(abc.ABC):
    """
    An excitation (a signal pulse) to transmit.
    """
    pass


@dataclass(frozen=True)
class SineWave(Pulse):
    """
    Sine wave excitation.

    :param frequency: transmitted carrier/nominal/center frequency [Hz]
    :param n_periods: number of sine periods in the transmitted burst, can be fractional
    :param inverse: whether the resulting wave should be inverted
    """
    center_frequency: float
    n_periods: float
    inverse: bool


class Aperture(abc.ABC):
    """
    An aperture - set of channels to perform.

    This class is abstract and should not be instantiated.
    """
    @abc.abstractmethod
    def get_size(self):
        pass


@dataclass(frozen=True)
class RegionBasedAperture(Aperture):
    """
    A region-based aperture.

    The aperture represents a single, contiguous range of channels.

    :param origin: an origin channel of the aperture
    :param size: a length of the aperture
    """
    origin: int
    size: int

    def get_size(self):
        return self.size


@dataclass(frozen=True)
class MaskAperture(Aperture):
    """
    A mask-based aperture.

    This aperture can represent a (possibly non-contiguous) set of channels,
    for example: ``[1,1,0,0,1,1]`` represents channels 0, 1, 4, 5

    :param mask: a mask to set, one dimensional numpy array
    """

    mask: np.ndarray

    def get_size(self):
        return np.sum(self.mask.astype(bool))


@dataclass(frozen=True)
class SingleElementAperture(Aperture):
    """
    Represents an aperture consisting of a single element.

    :param mask: an element of the aperture
    """
    element: int

    def get_size(self):
        return 1



