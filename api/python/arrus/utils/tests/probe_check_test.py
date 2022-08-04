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

    def _generate_random_signal(self):
        return np.random.random(
            (self.nframe, self.ntx, self.nsamp, self.nrx))

    def _generate_zeros_signal(self):
        return np.zeros(
            (self.nframe, self.ntx, self.nsamp, self.nrx))

    def _generate_ones_signal(self):
        return np.ones(
            (self.nframe, self.ntx, self.nsamp, self.nrx))

    def _put_fast_sine_into_signal_array(self, signal, value, nvalues):
        nframe, ntx, nsamp, nrx = signal.shape
        sample0 = 64
        for iframe in range(nframe):
            for itx in range(ntx):
                for isamp in range(nvalues):
                    for irx in range(nrx):
                        if np.mod(isamp, 2) == 0:
                            signal[iframe, itx, sample0+isamp, irx] = \
                                value
                        else:
                            signal[iframe, itx, sample0+isamp, irx] = \
                                -value
        return signal


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

    def test_extract_zeros_signal(self):
        # generate synthetic rf signal
        signal = self._generate_zeros_signal()
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

    def test_extract_zeros_signal(self):
        # generate synthetic rf signal
        signal = self._generate_zeros_signal()
        extracted = self.extractor.extract(signal)
        self.assertTrue(
            np.all(
                extracted == np.zeros(self.ntx)
            )
        )

    def test_signal_normalization(self):
        # generate synthetic rf signal
        signal = self._generate_random_signal()
        signal2 = signal*2
        extracted = self.extractor.extract(signal)
        extracted2 = self.extractor.extract(signal2)
        self.assertTrue(
            all(extracted == extracted2)
        )

    def test_extract_ones_signal(self):
        # generate synthetic rf signal
        signal = self._generate_ones_signal()
        signal2 = signal.copy()
        nframe, ntx, nsamp, nrx = signal.shape
        for iframe in range(int(np.ceil(nframe/2))):
            for itx in range(int(np.ceil(ntx/2))):
                for isamp in range(nsamp):
                    for irx in range(nrx):
                        signal[iframe, itx, isamp, irx] = 0
        extracted = self.extractor.extract(signal)
        extracted2 = self.extractor.extract(signal2)
        self.assertEqual(
            2*np.sum(extracted), np.sum(extracted2)
        )

    def test_extract_doubled_energy(self):
        signal = self._generate_zeros_signal()
        signal_short = self._put_fast_sine_into_signal_array(
            signal, value=1, nvalues=32)
        signal = self._generate_zeros_signal()            
        signal_long = self._put_fast_sine_into_signal_array(
            signal, value=1, nvalues=64)
        extracted_short = self.extractor.extract(signal_short)
        extracted_long = self.extractor.extract(signal_long)
        self.assertAlmostEqual(
            2*np.sum(extracted_short), np.sum(extracted_long), 1
        )

class SignalDurationTimeExtractorTest(AbstractExtractorTest):

    nrx = 192
    ntx = 16
    nframe = 1
    nsamp = 256
    extractor = SignalDurationTimeExtractor()
    
    def test_extract(self):
        signal = self._generate_zeros_signal()
        nframe, ntx, nsamp, nrx = signal.shape
        for iframe in range(nframe):
            for itx in range(ntx):
                for irx in range(nrx):
                    signal[iframe, itx, 100:116, irx] = 1
        extracted = self.extractor.extract(signal)
        self.assertTrue(
            all(extracted[0] == extracted)
            and extracted[0] >= 40
            and extracted[0] < 45
        )

    def test_extract_doubled_time(self):
        signal = self._generate_zeros_signal()
        signal_short = self._put_fast_sine_into_signal_array(
            signal, value=1, nvalues=8)
        signal = self._generate_zeros_signal()            
        signal_long = self._put_fast_sine_into_signal_array(
            signal, value=1, nvalues=16)
        extracted_short = self.extractor.extract(signal_short)
        extracted_long = self.extractor.extract(signal_long)
        self.assertAlmostEqual(
            2*np.sum(extracted_short), np.sum(extracted_long), 3
        )

# TODO: test_check_probe_data() ?
class ProbeHealthVerifierTest(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
