import unittest
import numpy as np
import scipy.io
import arrus.tests.tools as tools
import arrus.utils.parameters
from arrus.utils.imaging import reconstruct_rf_img, compute_tx_delays

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
            desired=np.zeros((len(self.z_grid), len(self.x_grid)))
        )

    def test_single_input_channel(self):
        # Given:
        rf = np.zeros((1024, 1, 1), dtype=np.int16)

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
            desired=np.zeros((len(self.z_grid), len(self.x_grid)))
        )


class ReconstructRfImgSimulatedDataTest(unittest.TestCase):

    def test_pwi_reconstruct(self):
        # Given:
        file_path = tools.get_dataset_path("bmode/rfPwi_field_v2.mat")
        rf, sys_parameters, acq_parameters = \
            arrus.utils.parameters.load_matlab_file(file_path)

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


class ComputeTxDelaysEdgeCaseTest(unittest.TestCase):
    """
    Tests edge cases for computing delays function.
    """
# (angles, focus, pitch, c=1490, n_chanels=128
    def setUp(self):
        # Default parameter values.
        self.angles = [-5, 0, 5]
        self.focus = 50e-3
        self.pitch = 0.3e-3
        self.c = 1490
        self.n_chanels = 128

    def test_single_plane_wave(self):
        # Given:
        focus = []
        angles = 0

        actual = compute_tx_delays(
            angles,
            focus,
            self.pitch,
            self.c,
            self.n_chanels
        )

        # Should be:
        # We expect an array of all zeros (no input signal, no output signal).
        np.testing.assert_equal(
            actual=actual,
            desired=np.zeros((1, 128), dtype=np.int16)
        )

    def test_positive_focus(self):
        # Given:
        focus = 1
        angles = 0
        pitch = 1
        c = 1,
        n_chan = 3

        actual = compute_tx_delays(
            angles,
            focus,
            pitch,
            c,
            n_chan
        )

        # Should be:
        # We expect an array of all zeros (no input signal, no output signal).
        np.testing.assert_almost_equal(
            actual=actual,
            desired=np.array([[0, np.sqrt(2)-1, 0]]),
            decimal=15
        )


    def test_negative_focus(self):
        # Given:
        focus = -1
        angles = 0
        pitch = 1
        c = 1,
        n_chan = 3

        actual = compute_tx_delays(
            angles,
            focus,
            pitch,
            c,
            n_chan
        )

        # Should be:
        # We expect an array of all zeros (no input signal, no output signal).
        np.testing.assert_almost_equal(
            actual=actual,
            desired=np.array([[np.sqrt(2)-1, 0, np.sqrt(2)-1]]),
            decimal=15
        )

    def test_triple_plane_wave(self):
        # Given:
        focus = []
        angles = [-5, 0, 5]
        pitch = self.pitch
        c = self.c
        nchan = self.n_chanels

        actual = compute_tx_delays(
            angles,
            focus,
            self.pitch,
            self.c,
            nchan
        )

        desired = np.empty([3, nchan])
        desired[0, :] = np.linspace(-(nchan-1), 0, nchan)*pitch/c*np.sin(angles[0])
        desired[1, :] = np.zeros((1, nchan))
        desired[2, :] = np.linspace(0, (nchan-1), nchan)*pitch/c*np.sin(angles[2])

        # Should be:
        # We expect an array of all zeros (no input signal, no output signal).
        np.testing.assert_almost_equal(
            actual=actual,
            desired=desired,
            decimal=15
        )


if __name__ == "__main__":
    unittest.main()
