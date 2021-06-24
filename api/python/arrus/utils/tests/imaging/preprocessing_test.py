import unittest
import numpy as np
from arrus.utils.tests.utils import ArrusImagingTestCase
from arrus.utils.imaging import (
    QuadratureDemodulation,
)


class QuadratureDemodulationTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = QuadratureDemodulation
        self.context = self.get_default_context()

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
        return np.squeeze(result)

    # Corner cases:
    def test_no_input_signal(self):
        """Empty input array should not be accepted. """
        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
            self.run_op(data=[])

    def test_0(self):
        # Given
        data = [0]
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, 0j)

    def test_1(self):
        # Given
        data = [1]
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, 2+0j)

    def test_m1(self):
        # Given
        data = [-1]
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, -2+0j)

    def test_1D(self):
        ''' Test use vector data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
             + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, expected)

    def test_2D(self):
        ''' Test use 2D array data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (10,2))

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, expected)

    def test_3D(self):
        ''' Test use 3D array data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13,11,3))
        print(data.shape)

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
