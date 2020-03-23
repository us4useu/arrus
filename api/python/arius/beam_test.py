import math
import unittest

import numpy as np

from arius.test_tools import mock_import


# Module mocks.
# TODO(pjarosik) beam module should not depend on device modules
class AriusMock:
    pass

class DBARLiteMock:
    pass

class HV256Mock:
    pass

mock_import(
    "arius.devices.iarius",
    Arius=AriusMock,
)
mock_import(
    "arius.devices.idbarLite",
    DBARLite=DBARLiteMock,
)
mock_import(
    "arius.devices.ihv256",
    HV256=HV256Mock
)

import arius.beam as _beam

class PlaneWaveProfileBuilderTest(unittest.TestCase):

    def test_compute_delays_angle_45(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=math.pi/4,
            speed_of_sound=1,
            pitch=1,
            aperture_size=3
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=[0, 1, 2]
        )

    def test_compute_delays_angle_minus_45(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=-math.pi/4,
            speed_of_sound=1,
            pitch=1,
            aperture_size=3
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=[2, 1, 0]
        )

    def test_compute_delays_angle_30(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=math.pi/6,
            speed_of_sound=1,
            pitch=1,
            aperture_size=4
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=[0, math.sqrt(3)/3, 2*math.sqrt(3)/3, 3*math.sqrt(3)/3]
        )

    def test_compute_delays_angle_minus_30(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=-math.pi/6,
            speed_of_sound=1,
            pitch=1,
            aperture_size=4
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=[3*math.sqrt(3)/3, 2*math.sqrt(3)/3, math.sqrt(3)/3, 0]
        )

    def test_compute_delays_angle_45_speed_of_sound_05(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=math.pi/4,
            speed_of_sound=0.5,
            pitch=1,
            aperture_size=3
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=[0, 2, 4]
        )

    def test_compute_delays_angle_45_pitch_3(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=math.pi/4,
            speed_of_sound=1,
            pitch=3,
            aperture_size=3
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=[0, 3, 6]
        )

    def test_compute_delays_angle_45_aperture_size_192(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=math.pi/4,
            speed_of_sound=1,
            pitch=1,
            aperture_size=192
        )
        np.testing.assert_almost_equal(
            actual=delays,
            desired=np.arange(0, 192)
        )

    def test_compute_delays_angle_35deg_aperture_size_192(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=math.pi/180*3.5,
            speed_of_sound=1450,
            pitch=0.245e-3,
            aperture_size=192
        )
        print(delays)
        # np.testing.assert_almost_equal(
        #     actual=delays,
        #     desired=np.arange(0, 192)
        # )

    def test_compute_delays_angle_m35deg_aperture_size_192(self):
        delays = _beam.PlaneWaveProfileBuilder.compute_delays(
            angle=-math.pi/180*3.5,
            speed_of_sound=1450,
            pitch=0.245e-3,
            aperture_size=192
        )
        print(delays)
        # np.testing.assert_almost_equal(
        #     actual=delays,
        #     desired=np.arange(0, 192)
        # )


if __name__ == "__main__":
    unittest.main()