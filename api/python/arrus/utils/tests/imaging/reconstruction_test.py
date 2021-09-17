import unittest
import numpy as np

from arrus.utils.tests.utils import ArrusImagingTestCase
from arrus.utils.imaging import (
    ReconstructLri,
    RxBeamforming)






class PwiReconstrutionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = ReconstructLri
        self.context = self.get_default_context()
        self.x_grid = np.linspace(-3*1e-3, 3*1e-3, 8)
        self.z_grid = np.linspace(9.5*1e-3, 11.*1e-3, 8)

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        #print(result)
        return np.squeeze(result)

    # Corner cases:
#    def test_no_input_signal(self):
#        """Empty input array should not be accepted. """
#        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
#            #pass
#        #    self.run_op(data=[], x_grid=[], z_grid=[])
#            self.run_op(data=[], x_grid=self.x_grid, z_grid=self.z_grid)



    def test_empty(self):
        # Given
        data = []
        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        print(expected_shape)
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)

    def test_0(self):
        # Given
        data = 0
        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        print(expected_shape)
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)






    # def test_1(self):
    #     # Given
    #     data = [1]
    #     # Run
    #     result = self.run_op(data=data)
    #     # Expect
    #     expected = 2+0j
    #     np.testing.assert_equal(result, expected)

    # def test_negative1(self):
    #     # Given
    #     data = [-1]
    #     # Run
    #     result = self.run_op(data=data)
    #     # Expect
    #     expected = -2+0j
    #     np.testing.assert_equal(result, expected)

    # def test_1D(self):
    #     ''' Test uses vector data.'''

    #     # Given
    #     data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)

    #     # Run
    #     result = self.run_op(data=data)

    #     # Expect
    #     fs = self.context.device.sampling_frequency
    #     fc = self.context.sequence.pulse.center_frequency
    #     n_samples = np.shape(data)[-1]
    #     t = (np.arange(0, n_samples) / fs)
    #     m = (  2 * np.cos(-2 * np.pi * fc * t)
    #            + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
    #     m = m.astype(np.complex64)
    #     expected = np.squeeze(m * data)
    #     np.testing.assert_equal(result, expected)

    # def test_2D(self):
    #     ''' Test uses 2D array data.'''

    #     # Given
    #     data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
    #     data = np.tile(data, (10, 2))

    #     # Run
    #     result = self.run_op(data=data)

    #     # Expect
    #     fs = self.context.device.sampling_frequency
    #     fc = self.context.sequence.pulse.center_frequency
    #     n_samples = np.shape(data)[-1]
    #     t = (np.arange(0, n_samples) / fs)
    #     m = (  2 * np.cos(-2 * np.pi * fc * t)
    #            + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
    #     m = m.astype(np.complex64)
    #     expected = np.squeeze(m * data)
    #     np.testing.assert_equal(result, expected)

    # def test_3D(self):
    #     ''' Test uses 3D array data.'''

    #     # Given
    #     data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
    #     data = np.tile(data, (13, 11, 3))

    #     # Run
    #     result = self.run_op(data=data)

    #     # Expect
    #     fs = self.context.device.sampling_frequency
    #     fc = self.context.sequence.pulse.center_frequency
    #     n_samples = np.shape(data)[-1]
    #     t = (np.arange(0, n_samples) / fs)
    #     m = (  2 * np.cos(-2 * np.pi * fc * t)
    #            + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
    #     m = m.astype(np.complex64)
    #     expected = np.squeeze(m * data)
    #     np.testing.assert_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

