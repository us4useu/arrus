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
from arrus.metadata import Metadata


# number of samples skipped at the beggining 
_N_SKIPPED_SAMPLES = 75

LOGGER = arrus.logging.get_logger()


def _get_mid_rx(nrx):
    """
    Returns channel in the receiving aperture
    corresponding to the transmit one.
    """
    return int(np.ceil(nrx/2) - 1)


def _hpfilter(
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


def _normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalizes input np.ndarray (i.e. moves values into [0, 1] range.
    If x contains some np.nans, they are ignored.
    If x contains only np.nans, non-modified x is returned.

    :param x: np.ndarray
    :return: normalized np.ndarray
    """
    mx = np.nanmax(x)
    if np.isfinite(mx):
        mn = np.nanmin(x)
        if mx != mn:
            normalized = (x - mn) / (mx - mn)
        else:
            normalized = 0
    else:
        normalized = x

    return normalized


def _envelope(rf: np.ndarray) -> np.ndarray:
    """
    Returns _envelope of the signal using Hilbert transform.

    :param rf: signals in np.ndarray
    :return: _envelope in np.ndarray
    """
    return np.abs(hilbert(rf))


class StdoutLogger:

    def __init__(self):
        for func in ("debug", "info", "error", "warning", "warn"):
            setattr(self, func, self.log)

    def log(self, msg):
        print(msg)


@dataclasses.dataclass(frozen=True)
class Footprint:
    """
    Contains footprint data - a reference signals and
    corresponding metadata.

    :param rf: np.ndarray of rf signals
    :param metadata: arrus metadata
    :param masked channels: list of channels masked during footprint acquisition
    :param timestamp: time of footprint creation in nanoseconds since epoch
                  (see time.time_ns() description)
    """
    rf: np.ndarray
    metadata: Metadata
    masked_elements: tuple
    timestamp: int

    def get_number_of_frames(self):
        return self.rf.shape[0]

    def get_tx_frequency(self):
        return self.metadata.context.sequence.pulse.center_frequency

    def get_sequence(self):
        return self.metadata.context.sequence


@dataclasses.dataclass(frozen=True)
class FeatureDescriptor:
    """
    Descriptor class for signal features used for probe 'diagnosis'.

    :param name: feature name ("amplitude" or "signal_duration_time")
    :param active_range: feature range of values possible to obtain from active
       'healthy' transducer
    :param masked_elements_range: feature range of values possible to obtain
       from inactive 'healthy' transducer
    """
    name: str
    active_range: tuple
    masked_elements_range: tuple


class ElementValidationVerdict(enum.Enum):
    """
    Contains element validation verdict.
    """
    VALID = enum.auto()
    TOO_HIGH = enum.auto()
    TOO_LOW = enum.auto()
    INDEFINITE = enum.auto()


@dataclasses.dataclass(frozen=True)
class ProbeElementValidatorResult:
    """
    Contains single element validation result.

    :param verdict: ElementValidationVerdict object
    :param valid_range: tuple contained valid range for examined feature
    """
    verdict: ElementValidationVerdict
    valid_range: tuple


@dataclasses.dataclass(frozen=True)
class ProbeElementFeatureDescriptor:
    """
    Descriptor class for results of element checking.

    :param name: name of the feature used for element check
    :param value: value of the feature
    :param validation_result: ProbeElementValidationResult object contained
        verdict and valid range for examined feature
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
    footprint: np.ndarray

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
    """
    Abstract class used for creation feature extractors.
    """
    feature: str

    @abstractmethod
    def extract(self, rf: np.ndarray, *args) -> np.ndarray:
        raise ValueError("Abstract class")


class MaxAmplitudeExtractor(ProbeElementFeatureExtractor):
    """
    Feature extractor class for extracting maximal amplitudes from array of
    rf signals.
    Returns vector of lenght equal to number of transmissions (ntx), where
    each element is a median over the frames of maximum amplitudes
    occurred in each of the tramissions.
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

    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Function extract parameter correlated with normalized signal energy.

        :param data: numpy array of rf data with the following shape:
            (number of frames, number of tx, number of samples,
            number of rx channels)
        :return: numpy array of signal energies
        """
        n_frames, ntx, _, nrx = data.shape
        energies = []
        for itx in range(ntx):
            frames_energies = []
            for frame in range(n_frames):
                rf = data[frame, itx, _N_SKIPPED_SAMPLES:, _get_mid_rx(nrx)]
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
        rf = _hpfilter(rf)
        rf = rf ** 2
        rf = _normalize(rf)
        return float(np.sum(rf))


class SignalDurationTimeExtractor(ProbeElementFeatureExtractor):
    """
    Feature extractor class for extracting signal duration times
    from array of rf signals.
    It is assumed that input data is acquired after very short excitation
    of a tested transducer.
    The pulse length (signal duration) is estimated via fitting gaussian
    function to _envelope of a high-pass filtered signal.
    """
    feature = "signal_duration_time"

    def extract(self, data: np.ndarray) -> np.array:
        """
        Extracts parameter correlated with signal duration time.

        :param data: numpy array of rf data with following dimensions:
        [number of repetitions,
         number of tx,
         number of samples,
         number of rx channels]
        :return: np.array of signal duration times
        """
        n_frames, ntx, _, nrx = data.shape
        times = []
        for itx in range(ntx):
            frames_times = []
            for iframe in range(n_frames):
                rf = data[iframe, itx, _N_SKIPPED_SAMPLES:, _get_mid_rx(nrx)]
                rf = rf.copy()
                rf = rf.astype(float)
                t = self.__get_signal_duration(np.squeeze(rf))
                frames_times.append(t)
            mean_time = np.mean(frames_times)
            times.append(mean_time)
        result = np.array(times)
        return result

    def __gauss(self, x: float, a: float, x0: float, sigma: float) -> float:
        """
        Returns the value of a gaussian function
            f(x)=a*exp(-(x-x0)**2/(2*sigma**2)
        at given argument x.

        :param x: argument
        :param a: height of the peak
        :param x0: expected value
        :param sigma: standard deviation
        :return: function value at x
        """
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
                # When curve_fit() can not fit gauss, sigma is set to 0
                pars = (0, 0, 0)
                LOGGER.log(arrus.logging.INFO,
                    "The expected signal _envelope couldn't be fitted "
                    "in some signal, probably due to low SNR.")

        return pars

    def __get_signal_duration(self, rf: np.ndarray) -> float:
        """
        Returns signal duration estimate.

        :param rf: signal vector
        :return: signal duration estimate in samples
        """
        rf = _hpfilter(rf)
        rf = _envelope(rf)
        # for return values, see definition of __gauss
        _, _, sigma = self.__fitgauss(rf)
        return round(3 * sigma)


class FootprintSimilarityPCCExtractor(ProbeElementFeatureExtractor):
    """
    Feature exctractor for extraction Pearson Correlation Coefficient (PCC)
    between given rf array and footprint rf array.
    """
    feature = "footprint_pcc"

    def extract(
            self,
            rf: np.ndarray,
            footprint_rf: np.ndarray
    )-> np.ndarray:

        smp = slice(_N_SKIPPED_SAMPLES, _N_SKIPPED_SAMPLES+256)
        crs = self.__get_corrcoefs(
            rf,
            footprint_rf,
            smp=smp,
        )
        return crs

    def __get_corrcoefs(
            self,
            rf,
            footprint_rf,
            smp=None,
            nround=3,
    ):
        if rf.shape != footprint_rf.shape:
            raise ValueError(
                "rf and footprint.rf arrays must have the same shape"
            )
        nframe, ntx, nsmp, nrx = rf.shape
        mid_rx = int(np.ceil(nrx/2) - 1)
        # average frames
        avdat = rf.mean(axis=0)
        avref = footprint_rf.mean(axis=0)
        crs = np.full(ntx, 0).astype(float)
        if smp is None:
            smp = slice(0, nsmp)
        for itx in range(ntx):
            dline = avdat[itx, smp, mid_rx]
            rline = avref[itx, smp, mid_rx]
            crs[itx] = np.corrcoef(dline, rline)[0,1].round(nround)
        # print(crs)
        return crs


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
    """
    Validator that check the value of the feature and compares it with
    given value range. When the value of the feature is within the given
    range the element is marked as VALID, otherwise it is marked as TOO_HIGH
    or TOO_LOW.
    """
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
                    EnergyExtractor,
                    FootprintSimilarityPCCExtractor]])


class ProbeHealthVerifier:
    """
    Probe health verifier class.
    """
    def get_footprint(
            self,
            cfg_path: str,
            n: int,
            tx_frequency: float,
            nrx: int=32,
            voltage: int=5,
    )-> Footprint:
        """
        Creates and returns Footprint object.
        """
        rfs, metadata, masked_elements = self._acquire_rf_data(
            cfg_path,
            n,
            tx_frequency,
            nrx,
            voltage,
        )
        footprint = Footprint(
            rf=rfs,
            metadata=metadata,
            masked_elements=masked_elements,
            timestamp=time.time_ns(),
        )
        return footprint

    def check_probe(
            self,
            cfg_path: str,
            n: int,
            tx_frequency: float,
            features: List[FeatureDescriptor],
            validator: ProbeElementValidator,
            nrx: int=32,
            voltage: int=5,
            footprint: Footprint=None,
    )-> ProbeHealthReport:
        """
        Checks probe elements by validating selected features
        of the acquired data.
        This metho:
        - runs data acquisition,
        - computes signal features,
        - tries to determine which elements are valid or not.

        :param cfg_path: a path to the system configuration file
        :param n: number of TX/RX sequences to execute (this may improve
                  feature value estimation)
        :param tx_frequency: pulse transmit frequency to be used in tx/rx scheme
        :param features: a list of features to check
        :param validator: ProbeElementValidator object, i.e. a validator
                          that should be used to determine if given parameter
                          have value within valid range
        :param nrx: size of the receiving aperture
        :param voltage: voltage to be used in tx/rx scheme
        :param footprint: object of the Footprint class;
                          if given, footprint tx/rx scheme will be used
        :return: an instance of the ProbeHealthReport
        """
        rfs, metadata, masked_elements = self._acquire_rf_data(
            cfg_path=cfg_path,
            n=n,
            tx_frequency=tx_frequency,
            nrx=nrx,
            voltage=voltage,
            footprint=footprint,
        )
        health_report = self._check_probe_data(
            rfs=rfs,
            footprint=footprint,
            metadata=metadata,
            masked_elements=masked_elements,
            features=features,
            validator=validator
        )
        return health_report

    def _check_probe_data(
            self,
            rfs: np.ndarray,
            metadata: arrus.metadata.ConstMetadata,
            footprint: Footprint,
            masked_elements: Set[int],
            features: List[FeatureDescriptor],
            validator: ProbeElementValidator
    ) -> ProbeHealthReport:
        """
        Creates probe health report.
        """
        _, ntx, _, _ = rfs.shape

        # Compute feature values, verify the values according to given
        # validator.
        results = {}
        for feature in features:
            extractor = EXTRACTORS[feature.name]()
            if feature.name == "footprint_pcc":
                try:
                    extractor_result = extractor.extract(rfs, footprint.rf)
                except:
                    raise ValueError(
                        "The footprint must by of a class Footprint. "
                        "Check if appropriate footprint is given."
                    )
            else:
                extractor_result = extractor.extract(rfs)

            validator_result = validator.validate(
                values=extractor_result,
                masked=masked_elements,
                active_range=feature.active_range,
                masked_range=feature.masked_elements_range
            )

            results[feature.name] = (extractor_result, validator_result)
            # print(extractor_result)

        masked_elements_set = set(masked_elements)
        # Prepare descriptor for examined element.
        elements_descriptors = []

        # For each examined channel
        for i in range(ntx):
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
                features=features
            ),
            elements=elements_descriptors,
            sequence_metadata=metadata,
            data=rfs,
            footprint=footprint
        )
        return report

    def _acquire_rf_data(
            self,
            cfg_path,
            n,
            tx_frequency,
            nrx,
            voltage,
            footprint=None,
    ):
        """
        Acquires rf data. If footprint is given the footprint sequence is used,
        """
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
            if footprint is None:
                seq = LinSequence(
                    tx_aperture_center_element=np.arange(0, n_elements),
                    tx_aperture_size=1,
                    tx_focus=30e-3,
                    pulse=Pulse(
                        center_frequency=tx_frequency,
                        n_periods=0.5,
                        inverse=False,
                    ),
                    rx_aperture_center_element=np.arange(0, n_elements),
                    rx_aperture_size=nrx,
                    rx_sample_range=(0, 512),
                    pri=1000e-6,
                    tgc_start=14,
                    tgc_slope=0,
                    downsampling_factor=1,
                    speed_of_sound=1490,
                )
            else:
                seq = footprint.get_sequence()
                n = footprint.get_number_of_frames()
                print("Sequence loaded from footprint.")

            scheme = Scheme(
                tx_rx_sequence=seq,
                processing=rf_reorder,
                work_mode="MANUAL",
            )
            buffer, const_metadata = sess.upload(scheme)
            rfs = []
            if voltage > 15:
                raise ValueError("The voltage can not be higher "
                                 "than 15V for probe check")
            us4r.set_hv_voltage(voltage)
            # Wait for the voltage to stabilize.
            time.sleep(1)
            # Start the device.
            LOGGER.log(arrus.logging.INFO, "Starting TX/RX")
            # Record RF frames.
            # Acquire n consecutive frames
            for i in range(n):
                LOGGER.log(arrus.logging.DEBUG, f"Performing TX/RX: {i}")
                sess.run()
                data = buffer.get()[0]
                rfs.append(np.squeeze(data.copy()))
            rfs = np.stack(rfs)
        return rfs, const_metadata, masked_elements
