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


class AbstractExtractorTest(unittest.TestCase):

    # nrx = 192
    # ntx = 16
    # nframe = 4
    # nsamp = 256
    # extractor = ProbeElementFeatureExtractor

    def _generate_random_signal(self):
        return np.random.random(
            (self.nframe, self.ntx, self.nsamp, self.nrx))

    def _generate_zero_signal(self):
        return np.zeros(
            (self.nframe, self.ntx, self.nsamp, self.nrx))

    # def test_extract_zero_signal(self):
        ### generate synthetic rf signal
        # signal = self._generate_zero_signal()
        # _, ntx, _, _ = signal.shape
        # extracted = self.extractor.extract(signal)
        # self.assertTrue(np.all(extracted == np.zeros(ntx)))


class MaxAmplitudeExtractorTest(AbstractExtractorTest):

    nrx = 192
    ntx = 16
    nframe = 8
    nsamp = 256
    max_amplitude = 100
    extractor = MaxAmplitudeExtractor()

    def test_extract_random_signal(self):
        # generate synthetic rf signal
        signal = self._generate_random_signal()
        # change some samples to max_amplitude
        for iframe in range(self.nframe):
            for itx in range(self.ntx):
                signal[iframe, itx, 64-itx, 128-iframe] = self.max_amplitude
        # check extractor on the generated signal
        extracted = self.extractor.extract(signal)
        self.assertTrue(
            np.all(
                extracted == np.ones(self.ntx)*self.max_amplitude
            )
        )

    def test_extract_zero_signal(self):
        # generate synthetic rf signal
        signal = self._generate_zero_signal()
        # check extractor on the generated signal
        extracted = self.extractor.extract(signal)
        self.assertTrue(
            np.all(
                extracted == np.zeros(self.ntx)
            )
        )




class EnergyExtractorTest(AbstractExtractorTest):

    nrx = 192
    ntx = 16
    nframe = 1
    nsamp = 256
    extractor = EnergyExtractor()
    # TODO: test z wygenerowanym sinusem o zadanej amplitudzie i dlugosci
    #       i co za tym idzie znanej energii (mniej wiecej)

    def test_extract_zero_signal(self):
        # generate synthetic rf signal
        signal = self._generate_zero_signal()
        extracted = self.extractor.extract(signal)
        self.assertTrue(
            np.all(
                extracted == np.zeros(self.ntx)
            )
        )


# TODO: 
class SignalDurationTimeExtractorTest(AbstractExtractorTest):
    pass
    
    
# TODO: test_check_probe_data() ?
class ProbeHealthVerifierTest(unittest.TestCase):
    pass

    
    

if __name__ == "__main__":
    unittest.main()
