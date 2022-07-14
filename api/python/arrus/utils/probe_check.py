import collections.abc
import dataclasses
import math
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt, hilbert

import arrus.logging
import arrus.utils.us4r
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse, Scheme
from arrus.utils.imaging import Pipeline, RemapToLogicalOrder

_N_SKIPPED_SAMPLES = 10
_NRX = 64
_MID_RX = int(np.ceil(_NRX / 2) - 1)
_N_SAMPLES = 1024

# possible verdicts
VALID = "VALID"
TOO_HIGH = "TOO_HIGH"
TOO_LOW = "TOO_LOW"
INDEFINITE = "INDEFINITE"
VALID_VERDICTS = {VALID, TOO_HIGH, TOO_LOW, INDEFINITE}

# possible features
AMPLITUDE = "amplitude"
DURATION = "signal duration time"
ENERGY = "energy"


class StdoutLogger:
    def __init__(self):
        for func in ("debug", "info", "error", "warning"):
            setattr(self, func, self.log)

    def log(self, msg):
        print(msg)


# TODO(zklog) all dataclasses here should be immutable ("frozen")
@dataclasses.dataclass()
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


@dataclasses.dataclass()
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
    valid_range: tuple
    verdict: str


class ProbeElementFeatureExtractor(ABC):
    feature: str

    @abstractmethod
    def extract(self, rf: np.ndarray) -> np.ndarray:
        pass


class MaxAmplitudeExtractor(ProbeElementFeatureExtractor):
    """
    Feature extractor class for extracting maximal amplitudes from array of
    rf signals.
    """
    feature = AMPLITUDE

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
    feature = ENERGY

    # TODO(zklog) use type hints for function parameters
    #  (see e.g. ProbeElementFeatureExtractor.extract)
    def extract(self, data):
        """
        Function extract parameter correlated with normalized signal energy.

        :param data: numpy array of rf data,
        :return: numpy array of signal energies.
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
                rf = rf.copy()
                rf = rf.astype(float)
                e = self.__get_signal_energy(np.squeeze(rf))
                frames_energies.append(e)
            mean_energy = np.mean(frames_energies)
            energies.append(mean_energy)
        return np.array(energies)

    def __hpfilter(self, rf, n=4, wn=1e5, fs=65e6):
        # TODO(zklog) use " instead of '
        btype = 'highpass'
        output = 'sos'
        # TODO(zklog) what the sos means here? speed of sound?
        sos = butter(n, wn, btype=btype, output=output, fs=fs)
        return sosfilt(sos, rf)

    def __normalize(self, x):
        mx = np.max(x)
        mn = np.min(x)
        return (x - mn) / (mx - mn)

    def __get_signal_energy(self, rf):
        # TODO(zklog) why are you copying RF data here?
        rf = rf.copy()
        rf = self.__hpfilter(rf)
        rf = rf ** 2
        rf = self.__normalize(rf)
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
    feature = DURATION

    def extract(self, data):
        # TODO(zklog) don't start each comment with "function", etc.
        # just write what this functions does
        # Start comments with upper case letters.
        """
        function extract parameter correlated with signal duration time.

        :param data: numpy array of rf data,
        :return: list, list of signal duration times
        """
        # TODO(zklog) why the below comment is not in the above docstring?
        # input data: (number of repetitions, number of tx, number of
        # samples, number of rx channels)
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

    def __gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def __hpfilter(self, rf, n=4, wn=1e5, fs=65e6):
        btype = 'highpass'
        output = 'sos'
        # TODO(zklog) what the sos means here? speed of sound?
        sos = butter(n, wn, btype=btype, output=output, fs=fs)
        return sosfilt(sos, rf)

    def __normalize(self, x):
        mx = np.max(x)
        mn = np.min(x)
        return (x - mn) / (mx - mn)

    def __envelope(self, rf):
        return np.abs(hilbert(rf))

    def __preprocess_rf(self, rf):
        """
        The function for initial preprocessing, before gauss curve fitting.
        preprocessing contains of highpass filtration and envelope detection.
        """
        # TODO(zklog) why are copying the data here?
        rf = rf.copy()
        rf = self.__hpfilter(rf)
        rf = self.__envelope(rf)
        return rf

    def __fitgauss(self, y):
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

    def __get_signal_duration(self, rf):
        rf = self.__preprocess_rf(rf)
        # for return values, see definition of __gauss
        _, _, sigma = self.__fitgauss(rf)
        return round(3 * sigma)


# A list of extractors
EXTRACTORS = {
    AMPLITUDE: MaxAmplitudeExtractor,
    DURATION: SignalDurationTimeExtractor,
    ENERGY: EnergyExtractor,
}


@dataclasses.dataclass(frozen=True)
class ProbeElementHealthReport:
    """
    Report of a single probe element health check.

    The probe element can be in one of the following states:
    - "VALID": the element seems to work correctly,
    - "TOO_HIGH": the element is characterised by too high feature value,
    - "TOO_LOW": the element is characterised by too low feature value,
    - "INDEFINITE": the estimate of the feature value failed on signal from
      the element.
    The information on feature value are in features attribute, where
    the list of ProbeElementFeatureDescriptor instances are stored.

    :param is_masked: whether the element was masked in the system cfg
    :param features: list of ProbeElementFeatureDescriptor objects
    :param element_number: element number
    """
    is_masked: bool
    features: dict
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
    elements: collections.abc.Iterable  # list of ProbeElementHealthReport
    data: np.ndarray


class ProbeHealthVerifier:
    """
    Probe health verifier class.
    """

    def __init__(self, log=None):
        self.log = log if log is not None else StdoutLogger()

    def check_probe(self, cfg_path: str, features: list, method: str, n=1,
                    subaperture_size=9, group_size=32):
        # TODO(zklog) keep max 80 characters per line
        """
        Checks probe elements by using selected method (current: 'neighborhood' or 'threshold').

        This method:
        - runs data acquisition,
        - computes probe characteristic (feature),
        - tries to determine which elements are valid or not.

        BEFORE CALLING THIS FUNCTION, PLEASE MAKE SURE THERE IS NO OTHER
        DAQ PROCESS ALREADY RUNNING.

        The invalid elements are determined in the following way:
        - for each probe element `i`:
          - if element `i` is masked: check if it's amplitude is within proper range.
            If it's not, mark the element with state TOO_HIGH or TOO_LOW.
            When feature value can not be estimated, the state is marked as INDEFINITE.
          - if element `i` is not masked and `neighborhood` method is selected:
            - first, determine the neighborhood of the element `i`.
              The neighborhood is determined by the `group_size` parameter,
              and consists of a given number of adjacent probe elements.
            - Then, estimate the expected feature value
              - exclude from the neighborhood all elements which have an
                amplitude below `inactive_threshold`, (that is they seem to be
                inactive, e.g. they were turned off using channels mask in the
                system configuration).
              - compute the median of the feature value in that neighborhood - this
                is our estimate of the expected amplitude.
              - Then determine if the element is valid, based on its feature value
                and expected `feature_range_in_neighborhood`,
                a pair `(feature_min, feature_max)`:
                - if its feature value is out of the range
                  [feature_min*median, feature_max*median]:
                  mark this element with state TOO_LOW, TOO_HIGH, or INDEFINITE
                - if its amplitude is above amplitude_max*center_amplitude:
                  otherwise mark the element with state "VALID".

        :param cfg_path: a path to the system configuration file,
        :param features: a list of feature descriptor instances,
        :param method: method used - `neighborhood` or `threshold`,
        :param n: number of TX/RX sequences to execute,
        :param amplitude_range: the accepted amplitude range, relative to the
          element's expected amplitude,
        :param subaperture_size: the size of the subaperture used in computing
          element's estimate amplitude (not used for now),
        :param group_size: the size of the neighborhood,
        :return: an instance of the ProbeHealthReport

        """
        rfs, metadata, masked_elements = self._acquire_rf_data(cfg_path, n)

        valid_methods = {"neighborhood": self._test_probe_elements_neighborhood,
                         "threshold": self._test_probe_elements_threshold}
        test_fun = valid_methods[method]
        reports = []
        # TODO a single report for all features
        for feature in features:
            extractor = EXTRACTORS[feature.name]()
            feature_values = extractor.extract(rfs)
            elements_report, values_near = test_fun(
                feature_values=feature_values,
                feature_descriptor=feature,
                masked_elements=masked_elements,
                group_size=group_size
            )
            report = ProbeHealthReport(
                params=dict(
                    method=method,
                    features=features,
                    subaperture_size=subaperture_size,
                    group_size=group_size),
                sequence_metadata=metadata,
                elements=elements_report,
                data=rfs
            )
            reports.append(report)
        return reports

    def _test_probe_elements_threshold(self, feature_values, feature_descriptor,
                                       masked_elements, group_size=None):
        element_reports = []
        masked_elements = set(masked_elements)
        for i, value in enumerate(feature_values):
            is_masked = i in masked_elements

            if is_masked:
                thr_min, thr_max = feature_descriptor.masked_elements_range
            else:
                thr_min, thr_max = feature_descriptor.active_range

            if value > thr_max:
                verdict = "TOO_HIGH"
            elif value < thr_min:
                verdict = "TOO_LOW"
            else:
                verdict = "VALID"

            element_reports.append(
                ProbeElementFeatureDescriptor(
                    name=
                )
                ProbeElementHealthReport(
                    is_masked=is_masked,
                    features=[
                        ProbeElementFeatureDescriptor(
                            name=feature_descriptor.name,
                            verdict=verdict,
                            value=value)
                    ],
                    element_number=i,
                )
            )
        return element_reports

    def _test_probe_elements_neighborhood(
            self,
            feature_values,
            feature_descriptor,
            masked_elements,
            group_size,
            feature_range_in_neighborhood=(0.5, 2),
            min_num_of_neighbors=5,
    ):
        if _N_ELEMENTS % group_size != 0:
            raise ValueError("Number of probe elements should be divisible by "
                             "group size.")
        masked_elements = set(masked_elements)

        # Generate report
        element_reports = []
        values_near = np.full((2, *feature_values.shape), np.nan)
        for i, value in enumerate(feature_values):
            is_masked = i in masked_elements
            if is_masked:
                # Masked elements should be below inactive threshold,
                # otherwise there is something wrong.
                thr_min, thr_max = feature_descriptor.masked_elements_range
                if value > thr_max:
                    verdict = "TOO_HIGH"
                elif value < thr_min:
                    verdict = "TOO_LOW"
                else:
                    verdict = "VALID"
            else:
                thr_min, thr_max = feature_descriptor.active_range

                if group_size == "all":
                    l, r = 0, _N_ELEMENTS
                else:
                    l = i - (math.ceil(group_size / 2) - 1)
                    l = max(l, 0)
                    r = i + group_size // 2 + 1
                    r = min(r, _N_ELEMENTS)
                near = feature_values[l:r]
                active_elements = np.argwhere(
                    np.logical_and(thr_min <= near, near <= thr_max))
                # TODO what if almost all near elements are inactive?
                # Exclude the current element.
                near = near[active_elements]
                num_of_neighbors = len(active_elements)

                if num_of_neighbors < min_num_of_neighbors:
                    verdict = INDEFINITE
                else:
                    try:
                        mn, mx = feature_range_in_neighborhood
                        center = np.median(near)
                        lower_bound = center * mn
                        upper_bound = center * mx
                        assert lower_bound <= upper_bound
                        values_near[0, i] = lower_bound
                        values_near[1, i] = upper_bound

                        if value > upper_bound:
                            verdict = TOO_HIGH
                            # self.log.warning(f'Element {i} : {verdict} {round((value - center) * 100 / center, 1)} % center={center}')
                        elif value < lower_bound:
                            verdict = TOO_LOW
                            # self.log.warning(f'Element {i} : {verdict} {round((center - value) * 100 / center, 1)} % center={center}')
                        else:
                            verdict = VALID
                    except:
                        verdict = INDEFINITE

            element_reports.append(
                ProbeElementHealthReport(
                    is_masked=is_masked,
                    features=[
                        ProbeElementFeatureDescriptor(
                            name=feature_descriptor.name,
                            verdict=verdict,
                            value=value)
                    ],
                    element_number=i,
                )
            )
        return element_reports, values_near

    def _acquire_rf_data(self, cfg_path, n):
        seq = LinSequence(
            tx_aperture_center_element=np.arange(0, _N_ELEMENTS),
            tx_aperture_size=1,
            tx_focus=30e-3,
            pulse=Pulse(center_frequency=8e6, n_periods=0.5, inverse=False),
            rx_aperture_center_element=np.arange(0, _N_ELEMENTS),
            rx_aperture_size=_NRX,
            rx_sample_range=(0, _N_SAMPLES),
            pri=1000e-6,
            tgc_start=14,
            tgc_slope=0,
            downsampling_factor=1,
            speed_of_sound=1490,
            sri=800e-3)

        with arrus.session.Session(cfg_path) as sess:
            rf_reorder = Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                ),
                placement="/GPU:0"
            )
            us4r = sess.get_device("/Us4R:0")
            masked_elements = us4r.channels_mask
            scheme = Scheme(tx_rx_sequence=seq, processing=rf_reorder,
                            work_mode="MANUAL")
            buffer, const_metadata = sess.upload(scheme)
            rfs = []
            reordered_rfs = []

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
            rfs = self._rflist2array(rfs)
        return rfs, const_metadata, masked_elements

    def _rflist2array(self, rflist):
        a = rflist[0]
        for i in range(1, len(rflist)):
            a = np.concatenate((a, rflist[i]))
        return a
