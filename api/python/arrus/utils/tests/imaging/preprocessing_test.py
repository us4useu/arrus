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

    # def test_no_input_signal(self):
    #     """Empty input array should not be accepted. """
    #     with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
    #         self.run_op(data=[])


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


    # def test_0(self):
    #     # Given
    #     data = [-1,0,1]
    #     # Run
    #     result = self.run_op(data=data)
    #     # Expect
    #     np.testing.assert_equal(result, [-2-0j, 0j, 2+0j])

if __name__ == "__main__":
    unittest.main()
