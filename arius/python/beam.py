from typing import List
import math
import numpy as np

import arius.python.device as _device
import arius.python.utils as _utils


class PlaneWaveProfileBuilder:
    def __init__(self):
        self.angle = None
        self.speed_of_sound = None
        self.pitch = None
        self.aperture_size = None

    def set_angle(self, angle: float):
        self.angle = angle

    def set_speed_of_sound(self, speed_of_sound: float):
        self.speed_of_sound = speed_of_sound

    def set_pitch(self, pitch: float):
        self.pitch = pitch

    def set_aperture_size(self, aperture_size: int):
        self.aperture_size = aperture_size

    def build(self) -> (_device.Subaperture, List[float]):
        """
        Returns Subaperture and delays according to builder attributes.

        :return: (subaperture, delays)
        """
        delays = self.compute(
            angle=self.angle,
            speed_of_sound=self.speed_of_sound,
            pitch=self.pitch,
            aperture_size=self.aperture_size
        )
        aperture = _device.Subaperture(0, self.aperture_size)
        return (aperture, delays)

    @staticmethod
    def compute_delays(angle: float, speed_of_sound: float, pitch: float, aperture_size: int):
        _utils.assert_not_none([
            (angle, "angle"),
            (speed_of_sound, "speed of sound"),
            (pitch, "pitch"),
            (aperture_size, "aperture size")
        ])
        dy = math.tan(angle)*pitch/speed_of_sound
        result = np.array([i*dy for i in range(aperture_size)])
        result = result-np.min(result)
        return result


def plane_wave(angle: float):
    builder = PlaneWaveProfileBuilder()
    builder.set_angle(angle)
    return builder

