import dataclasses
import numpy as np
import math


@dataclasses.dataclass(frozen=True)
class ProbeModel:
    n_elements: int
    pitch: float
    curvature_radius: float

    def __post_init__(self):
        element_pos_x, element_pos_z = self._compute_element_position()
        super().__setattr__("element_pos_x", element_pos_x)
        super().__setattr__("element_pos_z", element_pos_z)

    def _compute_element_position(self):
        # element position along the surface
        element_position = np.arange(-(self.n_elements - 1) / 2,
                                     (self.n_elements - 1) / 2)
        element_position = element_position * self.pitch

        if not self.is_convex_array():
            x_pos = element_position
            z_pos = np.zeros((self.n_elements, 1))
        else:
            angle = element_position / self.curvature_radius
            x_pos = self.curvature_radius * np.sin(angle)
            z_pos = self.curvature_radius * np.cos(angle)
            z_pos = z_pos - np.min(z_pos)
        return (x_pos, z_pos)

    def is_convex_array(self):
        return not (math.isnan(self.curvature_radius)
                    or self.curvature_radius == 0.0)


@dataclasses.dataclass(frozen=True)
class ProbeDTO:
    model: ProbeModel

    def get_id(self):
        return "Probe:0"