import unittest
import numpy as np

from arrus.utils.probe_check import *


class TestLogger:
    def info(self, msg):
        print(msg)

    def debug(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)


class AbstractElementValidatorTest(unittest.TestCase):

    def _get_masked_elements(self, report):
        return [(i, e) for i, e in enumerate(report) if e.is_masked]

    def _get_invalid_elements(self, report):
        return [(i, e) for i, e in enumerate(report)
                if e.verdict != ElementValidationVerdict.VALID]


class ByNeighborhoodValidatorTest(AbstractElementValidatorTest):

    def test_detects_value_drop_non_masked(self):
        validator = ByNeighborhoodValidator(
            group_size=32, feature_range_in_neighborhood=(0.5, 1.5),
            min_num_of_neighbors=5)
        signal = np.ones(192)*10000
        signal[50] = 100
        signal[100] = 500
        signal[125] *= 2
        report = validator.validate(
            values=signal, masked=(),
            active_range=(550, 30000), masked_range=(0, 550))
        invalid_elements = self._get_invalid_elements(report)
        self.assertEqual(invalid_elements[0][0], 50)
        self.assertEqual(invalid_elements[0][1].verdict,
                         ElementValidationVerdict.TOO_LOW)
        self.assertEqual(invalid_elements[1][0], 100)
        self.assertEqual(invalid_elements[1][1].verdict,
                         ElementValidationVerdict.TOO_LOW)
        self.assertEqual(invalid_elements[2][0], 125)
        self.assertEqual(invalid_elements[2][1].verdict,
                         ElementValidationVerdict.TOO_HIGH)

    def test_detects_amplitude_drop_masked(self):
        validator = ByNeighborhoodValidator(
            group_size=32, feature_range_in_neighborhood=(0.5, 1.5),
            min_num_of_neighbors=5)
        signal = np.ones(192)*10000
        signal[50] = 1
        signal[100] = 5
        signal[125] /= 3
        report = validator.validate(
            values=signal, masked=(50, 100),
            active_range=(4000, 20000), masked_range=(0, 550))
        invalid_elements = self._get_invalid_elements(report)
        print(invalid_elements)
        self.assertEqual(invalid_elements[0][0], 125)
        self.assertEqual(invalid_elements[0][1].verdict,
                         ElementValidationVerdict.TOO_LOW)
        # Note: 50 and 100 are fine, as they are masked


class ByThresholdValidatorTest(AbstractElementValidatorTest):

    def test_threshold_detects_amplitude_drop_non_masked(self):
        validator = ByThresholdValidator()
        signal = np.ones(192)*10000
        signal[50]  = 100
        signal[100] = 500
        signal[125] *= 3
        report = validator.validate(
            values=signal, masked=(),
            active_range=(4000, 20000), masked_range=(0, 550))
        invalid_elements = self._get_invalid_elements(report)
        self.assertEqual(len(invalid_elements), 3)
        self.assertEqual(invalid_elements[0][0], 50)
        self.assertEqual(invalid_elements[0][1].verdict,
                         ElementValidationVerdict.TOO_LOW)
        self.assertEqual(invalid_elements[1][0], 100)
        self.assertEqual(invalid_elements[1][1].verdict,
                         ElementValidationVerdict.TOO_LOW)
        self.assertEqual(invalid_elements[2][0], 125)
        self.assertEqual(invalid_elements[2][1].verdict,
                         ElementValidationVerdict.TOO_HIGH)


class MaxAmplitudeExtractorTest(unittest.TestCase):

    def test_extract(self):
        amplitude_extractor = MaxAmplitudeExtractor()
        nrx = 192
        ntx = 16
        nframe = 1
        nsamp = 256
        signal = np.random.random((nframe, ntx, nsamp, nrx))
        self.assertEqual(1,1)



if __name__ == "__main__":
    unittest.main()
