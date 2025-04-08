import dataclasses
import numpy as np
import math
from typing import Tuple, Optional

from arrus.devices.device import DeviceType, DeviceId

DEVICE_TYPE = DeviceType("Probe")


@dataclasses.dataclass(frozen=True)
class ProbeModelId:
    manufacturer: str
    name: str


@dataclasses.dataclass(frozen=True)
class Lens:
    """
    The lens applied on the surface of the probe.

    Currently, the model of the lens is quite basic and accustomed mostly to
    the linear array probes, e.g. we assume that the lens is dedicated to be
    focusing in the elevation direction.

    :param thickness: lens thickness measured at center of the elevation [m]
    :param speed_of_sound: the speed of sound in the lens material [m/s]
    :param focus: geometric elevation focus in water [m]
    """
    thickness: float
    speed_of_sound: float
    focus: Optional[float]


@dataclasses.dataclass(frozen=True)
class MatchingLayer:
    """
    The matching layer applied directly on the probe elements.

    :param thickness: matching layer thickness [m]
    :param speed_of_sound: matching layer speed of sound [m/s]
    """
    thickness: float
    speed_of_sound: float


@dataclasses.dataclass(frozen=True)
class ProbeModel:
    """
    Probe model.
    """
    model_id: ProbeModelId
    n_elements: int
    pitch: float
    curvature_radius: float
    tx_frequency_range: Tuple[float, float] = None
    lens: Optional[Lens] = None
    matching_layer: Optional[MatchingLayer] = None

    def __post_init__(self):
        element_pos_x, element_pos_z, element_angle = self._compute_element_position()
        super().__setattr__("element_pos_x", element_pos_x)
        super().__setattr__("element_pos_z", element_pos_z)
        super().__setattr__("element_angle", element_angle)

    def _compute_element_position(self):
        # element position along the surface
        element_position = np.arange(-(self.n_elements - 1) / 2,
                                     self.n_elements / 2)
        element_position = element_position * self.pitch

        if not self.is_convex_array():
            x_pos = element_position
            z_pos = np.zeros(self.n_elements)
            angle = np.zeros(self.n_elements)
        else:
            angle = element_position / self.curvature_radius
            x_pos = self.curvature_radius * np.sin(angle)
            z_pos = self.curvature_radius * np.cos(angle)
            z_pos = z_pos - np.min(z_pos)
        return x_pos, z_pos, angle

    @property
    def x_min(self):
        """
        A short to get the position of the "left-most" element.
        """
        return np.min(self.element_pos_x)

    @property
    def x_max(self):
        """
        A short to get the position of the "right-most" element.
        """
        return np.max(self.element_pos_x)

    def is_convex_array(self):
        return not (math.isnan(self.curvature_radius)
                    or self.curvature_radius == 0.0)


@dataclasses.dataclass(frozen=True)
class ProbeDTO:
    model: ProbeModel
    # TODO(0.12.0) make id obligatory (will break previous const metadata)
    device_id: DeviceId = DeviceId(DEVICE_TYPE, 0)
