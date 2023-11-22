from dataclasses import dataclass, field
import abc
import numpy as np
from typing import Sequence, Union, Optional
from numbers import Number


class Unit:
    """
    SI Unit definitions.
    """

    # base units
    s = "s" # second
    m = "m" # metre
    kg = "kg" # kilogram
    A = "A" # ampere
    K = "K" # kelvin
    mol = "mol" # mole
    cd = "cd" # candela

    # derived units
    rad = "rad" # radian
    sr = "sr" # steradian
    Hz = "Hz" # hertz
    N = "N" # newton
    Pa = "Pa" # pascal
    J = "J" # joule
    W = "W" # watt
    C = "C" # coulomb
    V = "V"  # volt
    O = "O"  # ohm
    S = "S"  # siemens
    Wb = "Wb"  # weber
    T = "T"  # tesla
    H = "H"  # henry
    Celsius = "Celsius"  # celsius
    lm = "lm"  # lumen
    lx = "lx"  # lux
    Bq = "Bq"  # becquerel
    Gy = "Gy"  # gray
    Sv = "Sv"  # sievert
    kat = "kat"  # katal

    # other units
    dB = "dB"  # decibel
    pixel = "pixel"
    mps = "m/s" # meter per second
    radps = "rad/s" # radian per second

class Space(abc.ABC):
    """
    Abstract class for parameter space definition.

    :param shape: shape of the parameter - n-d array (tuple of integer values)
    :param dtype: data type of the parameter, currently can be specified
                  as a numpy data type objects
    :param name: the name associated with each dimension. Should be a sequence of strings.
        This parameter can be used e.g. by the presentation layer to display a proper label
        for a given parameter. Can be None - in this case all dimensions will have
        no associated name. The provide sequence can have None values - in this case
        None value always mean "no name".
    :param unit: the unit associated with each dimension. This parameter can be used
        e.g. by the presentation layer to display a proper label for a given parameter.
        Can be None - in this case all dimensions will have
        no associated SI unit. The provide sequence can have None values - in this case
        None value always mean "no SI unit".
    """
    shape: Sequence[int] = field(default=None, init=False)
    dtype: object = field(default=np.float32, init=False)
    name: Optional[Sequence[str]] = field(default=None, init=False)
    unit: Optional[Sequence[Union[Unit, str]]] = field(default=None, init=False)

    def __init__(self, shape, dtype, name=None, unit=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.unit = unit

    def is_empty(self):
        """
        Returns true if this space is empty.
        """
        return len(self.shape) == 0

    def is_scalar(self):
        """
        Returns true when this space is a scalar, i.e. 0-order array.
        """
        return len(self.shape) == 1 and self.shape[0] == 1

    def is_vector(self):
        """
        Returns true when this space is a scalar, i.e. 1st-order array.
        """
        return len(self.shape) == 1 and self.shape[0] > 1


@dataclass
class Box(Space):
    """
    Continuous space defined by min/max (low/high) values of each dimension.
    The multidimensional region is closed on all dimensions (i.e. includes low/high values).

    The low/high parameters should have exactly the same length as the shape
    parameter

    :param low: a sequence of min values for each dimension
    :param high: a sequence of max values for each dimension
    """
    low: Sequence[Number]
    high: Sequence[Number]

    def __init__(self, low, high, shape, dtype, name=None, unit=None):
        super().__init__(shape, dtype, name, unit)
        self.low = low
        self.high = high


@dataclass
class ParameterDef:
    """
    Definition of the session parameter.

    :param name: a unique parameter name
    :param space: space of values the given parameter can take
    """
    name: str
    space: Space


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


