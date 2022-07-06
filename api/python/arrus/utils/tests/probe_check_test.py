import unittest
import numpy as np

from channel_mask_test import ProbeHealthVerifier

class TestLogger:
    def info(self, msg):
        print(msg)

    def debug(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)


class ProbeCheckTest(unittest.TestCase):

    def setUp(self):
        self.verifier = ProbeHealthVerifier(log=TestLogger())

    def test_detects_amplitude_drop_non_masked(self):
        signal = np.ones(192)*10000
        signal[50]  = 100
        signal[100] = 500
        signal[125] *= 2
        report = self.verifier.test_probe_elements_neighborhood(
            characteristic=signal, masked_elements=(),
            amplitude_range=(0.5, 1.5), inactive_threshold=550, group_size=32)
        invalid_elements = self._get_invalid_elements(report)
        self.assertEqual(invalid_elements[0][0], 50)
        self.assertEqual(invalid_elements[0][1].state, "TOO_LOW_AMPLITUDE")
        self.assertEqual(invalid_elements[1][0], 100)
        self.assertEqual(invalid_elements[1][1].state, "TOO_LOW_AMPLITUDE")
        self.assertEqual(invalid_elements[2][0], 125)
        self.assertEqual(invalid_elements[2][1].state, "TOO_HIGH_AMPLITUDE")

    def test_detects_amplitude_drop_masked(self):
        signal = np.ones(192)*10000
        signal[50]  = 100
        signal[100] = 500
        signal[125] /= 3
        report = self.verifier.test_probe_elements_neighborhood(
            characteristic=signal, masked_elements=(50, 100),
            amplitude_range=(0.5, 1.5), inactive_threshold=550, group_size=32)
        invalid_elements = self._get_invalid_elements(report)
        masked_elements = self._get_masked_elements(report)
        masked_elements = [i for i, e in masked_elements]
        masked_elements = set(masked_elements)
        self.assertEqual(invalid_elements[0][0], 125)
        self.assertEqual(invalid_elements[0][1].state, "TOO_LOW_AMPLITUDE")
        self.assertEqual(masked_elements, {50, 100})

    def test_threshold_detects_amplitude_drop_non_masked(self):
        signal = np.ones(192)*10000
        signal[50]  = 100
        signal[100] = 500
        signal[125] *= 3
        report = self.verifier.test_probe_elements_threshold(
            characteristic=signal, masked_elements=(),
            threshold=(4000, 20000)
        )
        invalid_elements = self._get_invalid_elements(report)
        self.assertEqual(len(invalid_elements), 3)
        self.assertEqual(invalid_elements[0][0], 50)
        self.assertEqual(invalid_elements[0][1].state, "TOO_LOW_AMPLITUDE")
        self.assertEqual(invalid_elements[1][0], 100)
        self.assertEqual(invalid_elements[1][1].state, "TOO_LOW_AMPLITUDE")
        self.assertEqual(invalid_elements[2][0], 125)
        self.assertEqual(invalid_elements[2][1].state, "TOO_HIGH_AMPLITUDE")

    def test_detects_unmasked_elements(self):
        data = np.load("tests/probe_check_test_magprobe_3_all.npy")
        characteristic = self.verifier.compute_characteristic(
            data, subaperture_size=9)
        expected = (15, 116, 173)
        report = self.verifier.test_probe_elements_neighborhood(
            characteristic=characteristic, masked_elements=expected,
            amplitude_range=(0.5, 1.5), inactive_threshold=550, group_size=32)
        actual = self._get_masked_elements(report)
        for i, e in actual:
            self.assertEqual(e.state, "TOO_HIGH_AMPLITUDE")

    def test_probe_elements_neighborhood_magprobe3_data_masked(self):
        data = np.load("tests/probe_check_test_magprobe_3_masked.npy")
        characteristic = self.verifier.compute_characteristic(
            data, subaperture_size=9)

        report = self.verifier.test_probe_elements_neighborhood(
            characteristic=characteristic, masked_elements=(15, 116, 173),
            amplitude_range=(0.5, 1.5), inactive_threshold=550, group_size=32)

        masked_elements = self._get_masked_elements(report)
        # All masked elements should be valid.
        for i, element in masked_elements:
            self.assertEqual(element.state, "VALID")
        # Get all invalid elements and compare with the expected one.
        invalid_elements = self._get_invalid_elements(report)
        invalid_elements_set = set((i for i, e in invalid_elements))
        self.assertEqual(invalid_elements_set, {160, 156, 117})


    def _get_masked_elements(self, report):
        return [(i, e) for i, e in enumerate(report) if e.is_masked]

    def _get_invalid_elements(self, report):
        return [(i, e) for i, e in enumerate(report) if e.state != "VALID"]


if __name__ == "__main__":
    unittest.main()