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
            data = data.astype(np.int16)
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    # Test cases:

    # Corner cases:

    def test_no_input_signal(self):
        """Empty input array should not be accepted. """
        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
            self.run_op(data=[])


    def test_simple_1d_convolution_single_coeff(self):
        """A simple and easy to analyse test case."""
        # Given
        data = np.arange(5).astype(np.int16)
        # Run
        result = self.run_op(data=data)
        print(result.shape)
        # Expect
        np.testing.assert_equal(result, data)

if __name__ == "__main__":
    unittest.main()
