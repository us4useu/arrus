import unittest
import numpy as np
import scipy.io
import arius.tests.tools as tools
import arius.utils.parameters
from arius.utils.imaging import reconstruct_rf_img


class ReconstructRfImgEdgeCaseTest(unittest.TestCase):
    """
    Tests edge cases for RF frame reconstruction.
    """

    def setUp(self):
        # Default parameter values.
        self.x_grid = np.linspace(-3*1e-3, 3*1e-3, 16)
        self.z_grid = np.linspace(9.5*1e-3, 11.*1e-3, 64)
        self.pitch = 0.245e-3
        self.fs = 40e-6
        self.fc = 5e-6
        self.c = 1540
        self.tx_aperture = 128
        self.tx_focus = None
        self.tx_angle = [0]
        self.tx_mode = 'pwi'
        self.n_pulse_periods = 0

    def test_no_input_signal(self):
        # Given:
        rf = np.zeros((1024, 128, 1), dtype=np.int16)

        actual = reconstruct_rf_img(
            rf,
            self.x_grid, self.z_grid,
            self.pitch, self.fs, self.fc, self.c,
            self.tx_aperture,
            self.tx_focus, self.tx_angle, self.n_pulse_periods,
            self.tx_mode)

        # Should be:
        # We expect an array of all zeros (no input signal, no output signal).
        np.testing.assert_equal(
            actual=actual,
            # TODO(zkLog) use len(self.x_grid), len(self.z_grid) instead
            desired=np.zeros((64, 16))
        )

    def test_single_input_channel(self):
        # Given:
        rf = np.zeros((1024, 37, 1), dtype=np.int16)

        actual = reconstruct_rf_img(
            rf,
            self.x_grid, self.z_grid,
            self.pitch, self.fs, self.fc, self.c,
            self.tx_aperture,
            self.tx_focus, self.tx_angle, self.n_pulse_periods,
            self.tx_mode)

        # Should be:
        np.testing.assert_equal(
            actual=actual,
            desired=np.zeros((64, 16))
        )
        # TODO(zkLog) this test will not pass:
        # I've assumed here all zeros, but we get NaNs
        # This will also contain NaNs for 16 or 24 channels.
        # Will work for 32 channels.


class ReconstructRfImgSimulatedDataTest(unittest.TestCase):

    def test_pwi_reconstruct(self):
        # Given:
        file_path = tools.get_dataset_path("bmode/rfPwi_field_v2.mat")
        rf, sys_parameters, acq_parameters = \
            arius.utils.parameters.load_matlab_file(file_path)

        desired = tools.get_dataset_path("bmode/rfPwi_field_v2_desired.mat")
        desired = scipy.io.loadmat(desired)['img']

        x_grid = np.linspace(-3*1e-3, 3*1e-3, 16)
        z_grid = np.linspace(9.5*1e-3, 11.*1e-3, 64)

        actual = reconstruct_rf_img(
            rf,
            x_grid=x_grid, z_grid=z_grid,
            pitch=sys_parameters.pitch,
            fs=acq_parameters.rx.sampling_frequency,
            fc=acq_parameters.tx.frequency,
            c=acq_parameters.speed_of_sound,
            tx_aperture=acq_parameters.tx.aperture_size,
            tx_focus=acq_parameters.tx.focus,
            tx_angle=acq_parameters.tx.angles,
            n_pulse_periods=acq_parameters.tx.n_periods,
            tx_mode=acq_parameters.mode)

        # Should be:
        np.testing.assert_equal(actual=actual, desired=desired)


if __name__ == "__main__":
    unittest.main()
