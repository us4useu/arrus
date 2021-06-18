import unittest

import numpy as np

from arrus.utils.tests.utils import ArrusImagingTestCase

from arrus.utils.imaging import (
    FirFilter,
    BandpassFilter
)


class FirFilterTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = FirFilter
        self.context = self.get_default_context()

    def run_op(self, **kwargs):
        # Currently FirFilter supports only 3D arrays,
        # the filtering is done along the last axis.
        # So we need to reshape the data to proper sizes.
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs['data'] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    # Test cases:

    # Corner cases:
    def test_no_input_signal(self):
        """Empty input array should not be accepted. """
        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
            self.run_op(data=[], taps=[])

    def test_simple_1d_convolution_single_coeff(self):
        """A simple and easy to analyse test case."""
        # Given
        data = np.arange(5)
        filter_taps = np.array([1])
        # Run
        result = self.run_op(data=data, taps=filter_taps)
        # Expect
        np.testing.assert_equal(result, data)

    def test_simple_1d_convolution(self):
        """A simple and easy to analyse test case, example 2."""
        data = np.arange(5)
        filter_taps = np.array([1, 1, 1])
        # Run
        result = self.run_op(data=data, taps=filter_taps)
        # Expect
        np.testing.assert_equal(result, [1, 3, 6, 9, 7])

    def test_convolution_properly_inversed(self):
        """ Test if the data is properly reversed in the convolution,
            i.e. sum(data[j-i] * coeffs[i]) """
        data = np.arange(5)
        filter_taps = np.array([1, 2, 3])
        # Run
        result = self.run_op(data=data, taps=filter_taps)
        # Expect
        #    [0, 1, 2, 3, 4]
        # 1. [2, 3]
        # 2. [1, 2, 3]
        # 3.    [1, 2, 3]
        # 4.       [1, 2, 3]
        # 5.          [1, 2]
        np.testing.assert_equal(result, [3, 8, 14, 20, 9])


if __name__ == "__main__":
    unittest.main()
