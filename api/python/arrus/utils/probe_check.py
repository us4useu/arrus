import collections.abc
import dataclasses
import enum
import math
import time
from abc import ABC, abstractmethod
from typing import Set, List, Iterable, Tuple, Dict

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt, hilbert

import arrus.session
import arrus.logging
import arrus.utils.us4r
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse, Scheme
from arrus.utils.imaging import Pipeline, RemapToLogicalOrder

LOGGER = arrus.logging.get_logger()

_N_SKIPPED_SAMPLES = 10
_NRX = 64
_MID_RX = int(np.ceil(_NRX / 2) - 1)


def hpfilter(
        rf: np.ndarray,
        n: int = 4,
        wn: float = 1e5,
        fs: float = 65e6
) -> np.ndarray:
    """
    Returns rf signals high-pass filtered using the Butterworth filter.

    :param rf: numpy array of rf signals
    :param n: the order of the iir filter
    :param wn: iir filter cut-off frequency
    :param fs: sampling frequency
    :return: numpy array of filtered rf signals
    """
    btype = "highpass"
    output = "sos"
    iir = butter(n, wn, btype=btype, output=output, fs=fs)
    return sosfilt(iir, rf)


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalizes input np.ndarray (i.e. moves values into [0, 1] range.
    :param x: np.ndarray
    :return: normalized np.ndarray
    """
    mx = np.max(x)
    mn = np.min(x)
    return (x - mn) / (mx - mn)


def envelope(rf: np.ndarray) -> np.ndarray:
    """
    Returns envelope of the signal using Hilbert transform.
    :param rf: signals in np.ndarray
    :return: envelope in np.ndarray
    """
    return np.abs(hilbert(rf))

class StdoutLogger:
    def __init__(self):
        for func in ("debug", "info", "error", "warning", "warn"):
            setattr(self, func, self.log)

    def log(self, msg):
        print(msg)


@dataclasses.dataclass(frozen=True)
class FeatureDescriptor:
    """
    Descriptor class for signal features used for probe 'diagnosis'.

    :param name: feature name ("amplitude" or "signal_duration_time"),
    :param active_range: feature range of values possible to obtain from active
       'healthy' transducer,
    :param masked_elements_range: feature range of values possible to obtain
       from inactive 'healthy' transducer.
    """
    name: str
    active_range: tuple
    masked_elements_range: tuple


class ElementValidationVerdict(enum.Enum):
    VALID = enum.auto()
    TOO_HIGH = enum.auto()
    TOO_LOW = enum.auto()
    INDEFINITE = enum.auto()


@dataclasses.dataclass(frozen=True)
class ProbeElementValidatorResult:
    verdict: ElementValidationVerdict
    valid_range: tuple


@dataclasses.dataclass(frozen=True)
class ProbeElementFeatureDescriptor:
    """
    Descriptor class for results of element checking.

    :param name: name of the feature used for element check,
    :param value: value of the feature.
    :param correct_range: range of values (min, max), for which the element will
      be marked as correct
    :param verdict: verdict string (one of following "VALID_VERDICTS")
    """
    name: str
    value: float
    validation_result: ProbeElementValidatorResult


@dataclasses.dataclass(frozen=True)
class ProbeElementHealthReport:
    """
    Report of a single probe element health check.

    The probe element can be in one of the following states:
    - "VALID": the element seems to work correctly,
    - "TOO_HIGH": the element is characterised by too high feature value,
    - "TOO_LOW": the element is characterised by too low feature value,
    - "INDEFINITE": the estimate of the feature value failed.
    The information on feature value are in features attribute, where
    the list of ProbeElementFeatureDescriptor instances are stored.

    :param is_masked: whether the element was masked in the system cfg
    :param features: dict of ProbeElementFeatureDescriptor oebjects
      [feature name -> feature descriptor]
    :param element_number: element number
    """
    is_masked: bool
    features: Dict[str, ProbeElementFeatureDescriptor]
    element_number: int


@dataclasses.dataclass(frozen=True)
class ProbeHealthReport:
    """
    A complete report of the probe health.

    Currently, the health report contains only information about the health
    of each probe element separately.

    :param params: a dictionary with health verifier method parameters
    :param sequence_metadata: description of the TX/RX sequence used in the
      probe health verification procedure
    :param elements: a list of `ProbeElementHealthReport` objects
    :param data: an RF data on the basis of which the probe verification was
        performed
    """
    # Report generator parameters.
    params: dict
    sequence_metadata: arrus.metadata.ConstMetadata
    # Report results
    elements: Iterable[ProbeElementHealthReport]
    data: np.ndarray

    @property
    def characteristics(self) -> Dict[str, np.ndarray]:
        result = collections.defaultdict(list)
        for e in self.elements:
            for name, desc in e.features.items():
                result[name].append(desc.value)

        # Convert ordinary lists to np.ndarrays
        for name, c in result.items():
            result[name] = np.asarray(c)
        return result


class ProbeElementFeatureExtractor(ABC):
    feature: str

    @abstractmethod
    def extract(self, rf: np.ndarray) -> np.ndarray:
        raise ValueError("Abstract class")


class MaxAmplitudeExtractor(ProbeElementFeatureExtractor):
    """
    Feature extractor class for extracting maximal amplitudes from array of
    rf signals.
    """
    feature = "amplitude"

    def extract(self, rf: np.ndarray) -> np.ndarray:
        # TODO(zklog) perhaps it might be a good idea to also remove the DC component here?
        rf = rf.copy()
        rf = np.abs(rf[:, :, _N_SKIPPED_SAMPLES:, :])
        # Reduce each RF frame into a vector of n elements
        # (where n is the number of probe elements).
        frame_max = np.max(rf[:, :, :, :], axis=(2, 3))
        # Choose median of a list of Tx/Rxs sequences.
        frame_max = np.median(frame_max, axis=0)
        return frame_max


class EnergyExtractor(ProbeElementFeatureExtractor):
    """
    Feature extractor class for extracting normalised signal energies
    from array of rf signals.
    It is assumed that input data is acquired after very short excitation
    of a tested transducer.
    The signal is normalised, thus when there is only noise,
    or the signal is long (ringing), the energy is high.
    """
    feature = "energy"

    # TODO(zklog) use type hints for function parameters
    #  (see e.g. ProbeElementFeatureExtractor.extract)
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Function extract parameter correlated with normalized signal energy.

        :param data: numpy array of rf data
        :return: numpy array of signal energies
        """
        # TODO(zklog) why the below comment is not in the above docstring?
        # input data: (number of repetitions, number of tx, number of
        # samples, number of rx channels)
        n_frames, ntx, _, nrx = data.shape
        energies = []
        for itx in range(ntx):
            frames_energies = []
            for frame in range(n_frames):
                rf = data[frame, itx, _N_SKIPPED_SAMPLES:, _MID_RX]
                rf = rf.astype(float)
                e = self.__get_signal_energy(np.squeeze(rf))
                frames_energies.append(e)
            mean_energy = np.mean(frames_energies)
            energies.append(mean_energy)
        return np.array(energies)


    def __get_signal_energy(self, rf: np.ndarray) -> np.float:
        """
        Returns normalized and high-pass filtered signal energy.
        :param rf: signal
        :return: signal energy (np.float)
        """
        rf = hpfilter(rf)
        rf = rf ** 2
        rf = normalize(rf)
        return np.sum(rf)


class SignalDurationTimeExtractor(ProbeElementFeatureExtractor):
    """
    Feature extractor class for extracting signal duration times
    from array of rf signals.
    It is assumed that input data is acquired after very short excitation
    of a tested transducer.
    The pulse length (signal duration) is estimated via fitting gaussian
    function to envelope of a high-pass filtered signal.
    """
    feature = "signal_duration_time"

    def __init__(self, log):
        self.log = log

    def extract(self, data: np.ndarray) -> list:
        """
        Extracts parameter correlated with signal duration time.

        :param data: numpy array of rf data with following dimensions:
        [number of repetitions,
         number of tx,
         number of samples,
         number of rx channels]
        :return: list, list of signal duration times
        """
        n_frames, ntx, _, nrx = data.shape
        times = []
        for itx in range(ntx):
            # todo number of repeats
            frames_times = []
            for frame in range(n_frames):
                # TODO(zklog) data[0 ???
                rf = data[0, itx, _N_SKIPPED_SAMPLES:, _MID_RX]
                rf = rf.copy()
                rf = rf.astype(float)
                t = self.__get_signal_duration(np.squeeze(rf))
                frames_times.append(t)
            mean_time = np.mean(frames_times)
            times.append(mean_time)
        result = np.array(times)
        return result

    def __gauss(self, x: float, a: float, x0: float, sigma: float) -> float:
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


    def __fitgauss(self, y: np.ndarray) -> tuple:
        """
        The function fits gauss curve to signal, and returns tuple of curve
        parameters.
        """
        if np.all(y == 0):
            pars = (0, 0, 0)
        else:
            bounds = (np.array([0, 0, 1]),
                      np.array([np.max(y), len(y), len(y) * 0.9]))
            p0 = [np.max(y) / 2, len(y) / 2, 20]
            try:
                pars, _ = curve_fit(self.__gauss, np.arange(y.size), y,
                                    bounds=bounds, p0=p0)
            except Exception as e:
                # TODO(zklog) use logger instead of print
                # When curve_fit() can not fit gauss, sigma is set to 0
                pars = (0, 0, 0)
                print("The expected signal envelope couldn't be fitted in some "
                      "signal, probably due to low SNR.")

        return pars

    def __get_signal_duration(self, rf: np.ndarray) -> float:
        rf = hpfilter(rf)
        rf = envelope(rf)
        # for return values, see definition of __gauss
        _, _, sigma = self.__fitgauss(rf)
        return round(3 * sigma)


class ProbeElementValidator(ABC):
    """
    Probe Element validator.

    Note: each probe element validator should store its public parameters
    as attributes with names that DOES NOT start with single underscore ("_").
    All private members should start with a single underscore.
    See the implementation of the "params" property for more details.
    """
    name: str

    @abstractmethod
    def validate(
            self,
            values: Iterable[float],
            masked: Iterable[int],
            active_range: Tuple[float, float],
            masked_range: Tuple[float, float]
    ) -> List[ProbeElementValidatorResult]:
        raise ValueError("Abstract method")

    @property
    def params(self) -> dict:
        return dict((k, v) for k, v in vars(self).items()
                    if not k.startswith("_"))


class ByThresholdValidator(ProbeElementValidator):
    name = "threshold"

    def __init__(self):
        pass

    def validate(
            self,
            values: Iterable[float],
            masked: Iterable[int],
            active_range: Tuple[float, float],
            masked_range: Tuple[float, float]
    ) -> List[ProbeElementValidatorResult]:
        LOGGER.log(arrus.logging.INFO, "Running validator by threshold.")
        result = []
        masked = set(masked)
        for i, value in enumerate(values):
            is_masked = i in masked
            if not is_masked:
                thr_min, thr_max = active_range
            else:
                thr_min, thr_max = masked_range

            if value > thr_max:
                verdict = ElementValidationVerdict.TOO_HIGH
            elif value < thr_min:
                verdict = ElementValidationVerdict.TOO_LOW
            else:
                verdict = ElementValidationVerdict.VALID
            result.append(ProbeElementValidatorResult(
                verdict=verdict,
                valid_range=(thr_min, thr_max)
            ))
        return result


class ByNeighborhoodValidator(ProbeElementValidator):
    """
    Validator that compares each element with its neighborhood.

    The invalid elements are determined in the following way:
        - for each probe element `i`:
          - if element `i` is masked: check if it's feature is within proper
            range. If it's not, mark the element with state TOO_HIGH or TOO_LOW.
          - if element `i` is not masked:
            - first, determine the neighborhood of the element `i`.
              The neighborhood is determined by the `group_size` parameter,
              and consists of a given number of adjacent probe elements.
            - Then, estimate the expected feature value:
              - exclude from the neighborhood all elements which have an
                amplitude outside the active_range
                (see FeatureDescriptor class), that is they seem to be inactive
                e.g. they were turned off using channels mask in the system
                configuration,
              - if the number of active elements is less than the
                min_num_of_neighbors, set verdict for the element to
                INDEFINITE,
              - otherwise: compute the median in the neighborhood - this is our
                estimate of the expected feature value,
              - Then determine if the element is valid, based on its feature
                value and the `feature_range_in_neighborhood` param, which
                should be equal to `(feature_min, feature_max)`:
                - if the value is out of the range
                  [feature_min*median, feature_max*median]:
                  mark this element with state TOO_LOW, TOO_HIGH,
                - if its amplitude is above amplitude_max*center_amplitude:
                  otherwise mark the element with state "VALID".

    :param group_size: number of elements in the group
    :param feature_range_in_neighborhood: pair (feature_min, feature_max)
      (see the description above)
    :param min_num_of_neighbors: minimum number of active neighborhood elements
      that can be used to estimate group's (lower, upper) bound of amplitude;
      if the number of active elements is < min_nim_of_neighbors, the verdict
      for the elements will be INDEFINITE
    """
    name = "neighborhood"

    def __init__(self, group_size=32, feature_range_in_neighborhood=(0.5, 2),
                 min_num_of_neighbors=5):
        self.group_size = group_size
        self.feature_range_in_neighborhood = feature_range_in_neighborhood
        self.min_num_of_neighbors = min_num_of_neighbors

    def validate(
            self,
            values: List[float],
            masked: Iterable[int],
            active_range: Tuple[float, float],
            masked_range: Tuple[float, float]
    ) -> List[ProbeElementValidatorResult]:

        n_elements = len(values)
        if n_elements % self.group_size != 0:
            raise ValueError(
                "Number of probe elements should be divisible by "
                "group size.")
        masked_elements = set(masked)

        # Generate report
        results = []

        for i, value in enumerate(values):
            is_masked = i in masked_elements

            lower_bound, upper_bound = None, None

            if is_masked:
                # Masked elements should be below inactive threshold,
                # otherwise there is something wrong.
                thr_min, thr_max = masked_range
                if value > thr_max:
                    verdict = ElementValidationVerdict.TOO_HIGH
                elif value < thr_min:
                    verdict = ElementValidationVerdict.TOO_LOW
                else:
                    verdict = ElementValidationVerdict.VALID
            else:
                thr_min, thr_max = active_range

                # get elements from range [l, r)
                if self.group_size == "all":
                    l, r = 0, n_elements
                else:
                    l = i - (math.ceil(self.group_size / 2) - 1)
                    l = max(l, 0)
                    r = i + self.group_size // 2 + 1
                    r = min(r, n_elements)

                near = values[l:r]
                active_elements = np.argwhere(
                    np.logical_and(thr_min <= near, near <= thr_max))
                # Exclude the current element.
                near = near[active_elements]
                num_of_neighbors = len(active_elements)

                if num_of_neighbors < self.min_num_of_neighbors:
                    verdict = ElementValidationVerdict.INDEFINITE
                else:
                    mn, mx = self.feature_range_in_neighborhood
                    center = np.median(near)
                    lower_bound = center * mn
                    upper_bound = center * mx
                    assert lower_bound <= upper_bound

                    if value > upper_bound:
                        verdict = ElementValidationVerdict.TOO_HIGH
                    elif value < lower_bound:
                        verdict = ElementValidationVerdict.TOO_LOW
                    else:
                        verdict = ElementValidationVerdict.VALID
            results.append(
                ProbeElementValidatorResult(
                    verdict=verdict,
                    valid_range=(lower_bound, upper_bound)))
        return results


EXTRACTORS = dict([(e.feature, e) for e in
                   [MaxAmplitudeExtractor,
                    SignalDurationTimeExtractor,
                    EnergyExtractor]])


class ProbeHealthVerifier:
    """
    Probe health verifier class.
    """

    def __init__(self, log=None):
        self.log = log if log is not None else LOGGER

    def check_probe(
            self,
            cfg_path: str, n: int,
            features: List[FeatureDescriptor],
            validator: ProbeElementValidator) -> ProbeHealthReport:
        """
        Checks probe elements by validating selected features of the acquired
        data.

        This method:
        - runs data acquisition,
        - computes signal features,
        - tries to determine which elements are valid or not.

        :param cfg_path: a path to the system configuration file,
        :param n: number of TX/RX sequences to execute (this may improve
          feature value estimation),
        :param features: a list of features to check,
        :param validator: ProbeElementValidator instance, i.e. a validator
          that should be used to determine
        :return: an instance of the ProbeHealthReport
        """
        rfs, metadata, masked_elements = self._acquire_rf_data(cfg_path, n)
        return self.check_probe_data(
            rfs=rfs, metadata=metadata,
            masked_elements=masked_elements,
            features=features, validator=validator)

    def check_probe_data(
            self,
            rfs: np.ndarray, metadata: arrus.metadata.ConstMetadata,
            masked_elements: Set[int],
            features: List[FeatureDescriptor],
            validator: ProbeElementValidator) -> ProbeHealthReport:
        n_seq, n_tx_channels, n_samples, n_rx = rfs.shape

        # Compute feature values, verify the values according to given
        # validator.
        results = {}
        for feature in features:
            extractor = EXTRACTORS[feature.name]
            extractor_result = extractor.extract(rfs)
            validator_result = validator.validate(
                values=extractor_result,
                masked=masked_elements,
                active_range=feature.active_range,
                masked_range=feature.masked_elements_range
            )
            results[feature.name] = (extractor_result, validator_result)

        # Prepare descriptor for examined element.
        masked_elements_set = set(masked_elements)
        elements_descriptors = []

        # For each examined channel
        for i in range(n_tx_channels):
            # For each examined feature
            feature_descriptors = {}
            for feature in features:
                extractor_result, validator_result = results[feature.name]
                feature_value = extractor_result[i]
                element_validation_result = validator_result[i]

                descriptor = ProbeElementFeatureDescriptor(
                    name=feature.name,
                    value=feature_value,
                    validation_result=element_validation_result
                )
                feature_descriptors[feature.name] = descriptor
            element_descriptor = ProbeElementHealthReport(
                is_masked=i in masked_elements_set,
                features=feature_descriptors,
                element_number=i
            )
            elements_descriptors.append(element_descriptor)

        report = ProbeHealthReport(
            params=dict(
                method=validator.name,
                method_params=validator.params,
                features=features),
            sequence_metadata=metadata,
            elements=elements_descriptors,
            data=rfs
        )
        return report

    def _acquire_rf_data(self, cfg_path, n):
        with arrus.session.Session(cfg_path) as sess:
            rf_reorder = Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                ),
                placement="/GPU:0"
            )
            us4r = sess.get_device("/Us4R:0")
            n_elements = us4r.get_probe_model().n_elements
            masked_elements = us4r.channels_mask
            seq = LinSequence(
                tx_aperture_center_element=np.arange(0, n_elements),
                tx_aperture_size=1,
                tx_focus=30e-3,
                pulse=Pulse(center_frequency=8e6, n_periods=0.5,
                            inverse=False),
                rx_aperture_center_element=np.arange(0, n_elements),
                rx_aperture_size=_NRX,
                rx_sample_range=(0, 1024),
                pri=1000e-6,
                tgc_start=14,
                tgc_slope=0,
                downsampling_factor=1,
                speed_of_sound=1490,
                sri=800e-3)
            scheme = Scheme(tx_rx_sequence=seq, processing=rf_reorder,
                            work_mode="MANUAL")
            buffer, const_metadata = sess.upload(scheme)
            rfs = []
            us4r.set_hv_voltage(5)
            # Wait for the voltage to stabilize.
            time.sleep(1)
            # Start the device.
            self.log.info("Starting TX/RX")
            # Record RF frames.
            # Acquire n consecutive frames
            for i in range(n):
                self.log.debug(f"Performing TX/RX: {i}")
                sess.run()
                data = buffer.get()[0]
                rfs.append(data.copy())
            rfs = np.stack(rfs)
        return rfs, const_metadata, masked_elements
