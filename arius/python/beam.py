from typing import List
import math
import numpy as np

import arius.python.device as _device
import arius.python.utils as _utils


class BeamProfileBuilder:
    def __init__(self):
        self.speed_of_sound = None
        self.pitch = None
        self.aperture_size = None

    def set_speed_of_sound(self, speed_of_sound: float):
        self.speed_of_sound = speed_of_sound
        return self

    def set_pitch(self, pitch: float):
        self.pitch = pitch
        return self

    def set_aperture_size(self, aperture_size: int):
        self.aperture_size = aperture_size
        return self

    def build(self):
        raise NotImplementedError


class PlaneWaveProfileBuilder(BeamProfileBuilder):
    def __init__(self):
        super().__init__()
        self.angle = None

    def set_angle(self, angle: float):
        self.angle = angle
        return self

    def build(self):
        """
        Returns Subaperture and delays according to builder attributes.

        :return: (subaperture, delays)
        """
        delays = self.compute_delays(
            angle=self.angle,
            speed_of_sound=self.speed_of_sound,
            pitch=self.pitch,
            aperture_size=self.aperture_size
        )
        aperture = _device.Subaperture(0, self.aperture_size)
        return (aperture, delays)

    @staticmethod
    def compute_delays(angle: float, speed_of_sound: float, pitch: float, aperture_size: int):
        """
        Returns array of delays according to given parameters.

        :param angle: plane wave angle [rad]
        :param speed_of_sound: assumed speed of sound [m/s]
        :param pitch: a distance between probe's elements [m]
        :param aperture_size: size of an aperture
        :return: array of delays [s]
        """
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


def plane_wave(angle: float, speed_of_sound: float=1540) -> PlaneWaveProfileBuilder:
    """
    Returns a plane wave builder, with partially applied angle.

    The return value should be provided as an input to appropriate device
    (i.e. a probe instance).

    :param speed_of_sound: assumed speed of sound
    :param angle: plane wave angle [rad]
    :return: plane wave builder with applied angle
    """
    builder = PlaneWaveProfileBuilder()
    builder.set_angle(angle)
    builder.set_speed_of_sound(speed_of_sound)
    return builder

