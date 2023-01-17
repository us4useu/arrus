import numpy as np
import argparse
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse, Scheme
import arrus.logging
import arrus.utils.us4r
import time
import pickle
import dataclasses
import math
import collections.abc

from arrus.utils.imaging import Pipeline, RemapToLogicalOrder

arrus.set_clog_level(arrus.logging.INFO)

_N_SKIPPED_SAMPLES = 10
_N_ELEMENTS = 192
_N_SAMPLES = 512


VALID = "VALID"
TOO_HIGH_AMPLITUDE = "TOO_HIGH_AMPLITUDE"
TOO_LOW_AMPLITUDE = "TOO_LOW_AMPLITUDE"


@dataclasses.dataclass(frozen=True)
class ProbeElementHealthReport:
    """
    Report of a single probe element health check.

    The probe element can be in one of the following states:
    - "VALID": the element seems to work correctly,
    - "TOO_HIGH_AMPLITUDE": the element seems too generate to high signal amplitude
    - "TOO_LOW_AMPLITUDE": the element seems too produce to low signal amplitude

    :param is_masked: whether the element was masked in the system cfg
    :param state: detected probe element
    :param amplitude: measured signal amplitude
    """
    is_masked: bool
    state: str
    amplitude: float


@dataclasses.dataclass(frozen=True)
class ProbeHealthReport:
    """
    A complete report of the probe health.

    Currently the health report contains only information about the
    the health of each probe element separately.

    :param params: a dictionary with health verifier method parameters
    :param sequence_metadata: description of the TX/RX sequence used in the
      probe health verification procedure
    :param elements: a list of `ProbeElementHealthReport` objects
    :param data: an RF data on the basis of which the probe verification was performed
    :param characteristic: characteristic based on the RF data, this characteristic
     was used to estimate the health of the probe elements.
    """
    # Report generator parameters.
    params: dict
    sequence_metadata: arrus.metadata.ConstMetadata
    # Report results
    elements: collections.abc.Iterable  # list of ProbeElementHealthReport
    data: np.ndarray
    characteristic: list


class ProbeHealthVerifier:
    def __init__(self, log):
        self.log = log

    def check_probe_by_neighborhood(self, cfg_path,
                                    n=1, amplitude_range=(0.6, 1.4),
                                    inactive_threshold=550, rx_aperture_size=64, subaperture_size=9,
                                    group_size=32):
        """
        Check probe elements by using method "neighborhood".

        This method:
        - runs data acquisition,
        - computes probe characteristic,
        - tries to determine which elements are valid or not.

        BEFORE CALLING THIS FUNCTION, PLEASE MAKE SURE THERE IS NO OTHER
        DAQ PROCESS ALREADY RUNNING.

        The invalid elements are determined in the following way:
        - for each probe element `i`:
          - if element `i` is masked: check if it's amplitude is below given
            `inactive_threshold`, if it's not, mark the element with state
             "TOO_HIGH_AMPLITUDE".
          - if element `i` is not masked:
            - first, determine the neighborhood of the element `i`.
              The neighborhood is determined by the `group_size` parameter,
              and consists of a given number of adjacent probe elements.
            - Then, estimate the expected amplitude value (`center_amplitude`)
               for that neighborhood:
              - exclude from the neighborhood all elements which have an
                amplitude below `inactive_threshold`, (that is they seems to be
                inactive, e.g. they were turned off using channels mask in the
                system configuration).
              - compute the median of the amplitude in that neighborhood - this
                is our estimate of the expected amplitude.
              - Then determine if the element is valid, based on it's amplitude
                and expected `amplitude_range`,
                a pair `(amplitude_min, amplitude_max)`:
                - if its amplitude is below amplitude_min*center_amplitude:
                  mark this element with state "TOO_LOW_AMPLITUDE"
                - if its amplitude is above amplitude_max*center_amplitude:
                  mark this element with state "TOO_HIGH_AMPLITUDE" otherwise
                  mark the element with state "VALID".

        :param cfg_path: a path to the system configuration file.
        :param n: Number of TX/RX sequences to execute.
        :param amplitude_range: the accepted amplitude range, relative to the
          element's expected amplitude.
        :param inactive_threshold: the threshold, the amplitude threshold below
          which the element will be considered inactive (e.g. masked in the cfg)
        :param rx_aperture_size: the number of receiving elements
        :param subaperture_size: the size of the subaperture used in computing
          element's estimate amplitude.
        :param group_size: the size of the neighborhood
        :return: an instance of the ProbeHealthReport
        """
        subaperture_size = min(subaperture_size, rx_aperture_size)
        rfs, const_metadata, masked_elements = self.acquire_rf_data(cfg_path, n, rx_aperture_size)
        characteristic = self.compute_characteristic(rfs, rx_aperture_size, subaperture_size)

        elements_report = self.test_probe_elements_neighborhood(
            characteristic=characteristic,
            masked_elements=masked_elements,
            amplitude_range=amplitude_range,
            inactive_threshold=inactive_threshold,
            group_size=group_size
        )
        return ProbeHealthReport(
            params=dict(
                method="neighborhood",
                amplitude_range=amplitude_range,
                inactive_threshold=inactive_threshold,
                subaperture_size=subaperture_size,
                group_size=group_size),
            sequence_metadata=const_metadata,
            elements=elements_report,
            data=rfs,
            characteristic=characteristic
        )

    def check_probe_by_threshold(self, cfg_path, n=1, rx_aperture_size=64, subaperture_size=9,
                                 threshold=(4000, 20000)):
        """
         Check probe elements by using method "threshold".

         This method:
         - runs data acquisition,
         - computes probe characteristic,
         - tries to determine which elements are valid or not.

         BEFORE CALLING THIS FUNCTION, PLEASE MAKE SURE THERE IS NO OTHER
         DAQ PROCESS ALREADY RUNNING.

         :param cfg_path: a path to the system configuration file.
         :param n: Number of TX/RX sequences to execute.
         :param threshold: a pair of (min, max) of accepted amplitudes
         :param rx_aperture_size: the number of receiving elements
         :param subaperture_size: the size of the subaperture used in computing
           element's estimate amplitude.
         :return: an instance of the ProbeHealthReport
         """
        rfs, const_metadata, masked_elements = self.acquire_rf_data(cfg_path, n, rx_aperture_size)
        characteristic = self.compute_characteristic(rfs, rx_aperture_size, subaperture_size)

        elements_report = self.test_probe_elements_threshold(
            characteristic=characteristic,
            masked_elements=masked_elements,
            threshold=threshold
        )
        return ProbeHealthReport(
            params=dict(method="threshold", threshold=threshold),
            sequence_metadata=const_metadata,
            elements=elements_report,
            data=rfs,
            characteristic=characteristic
        )

    def acquire_rf_data(self, cfg_path, n, rx_aperture_size):
        seq = LinSequence(
            tx_aperture_center_element=np.arange(0, _N_ELEMENTS),
            tx_aperture_size=1,
            tx_focus=30e-3,
            pulse=Pulse(center_frequency=7e6, n_periods=2, inverse=False),
            rx_aperture_center_element=np.arange(0, _N_ELEMENTS),
            rx_aperture_size=rx_aperture_size,
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
        return data, const_metadata, masked_elements

    def compute_characteristic(self, rf, rx_aperture_size, subaperture_size):
        # Clear samples 65-85.
        # Around the sample 65 there is a switch from
        # The 1us after after the last transmission is the moment when RX on AFE turns on.
        # At the time of switching to RX, the acquired signal can contain a noise,
        # the amplitude of which increases with the decrease in receiving aperture.
        # The range 65-85 was selected experimentally.
        rf[:, :, 65:85, :] = 0
        rf = np.abs(rf[:, :, _N_SKIPPED_SAMPLES:, :])
        # Reduce each RF frame into a vector of n elements
        # (where n is the number of probe elements).
        start_channel = int(np.ceil(rx_aperture_size/2)-1)-(math.ceil(subaperture_size/2)-1)
        end_channel = int(np.ceil(rx_aperture_size/2)-1)+subaperture_size//2+1
        frame_max = np.max(rf[:, :, :, start_channel:end_channel], axis=(2, 3))
        # Choose median of a list of Tx/Rxs sequences.
        frame_max = np.median(frame_max, axis=0)
        return frame_max

    def test_probe_elements_threshold(self, characteristic, masked_elements,
                                      threshold):
        result = []
        amp_min, amp_max = threshold
        masked_elements = set(masked_elements)
        for i, amplitude in enumerate(characteristic):
            is_masked = i in masked_elements
            if amplitude > amp_max:
                state = "TOO_HIGH_AMPLITUDE"
            elif amplitude < amp_min:
                state = "TOO_LOW_AMPLITUDE"
            else:
                state = "VALID"
            result.append(ProbeElementHealthReport(
                is_masked=is_masked, state=state, amplitude=amplitude
            ))
        return result

    def test_probe_elements_neighborhood(
            self, characteristic, masked_elements, amplitude_range,
            inactive_threshold, group_size):
        """
        """
        if _N_ELEMENTS % group_size != 0:
            raise ValueError("Number of probe elements should be divisible by "
                             "group size.")

        amp_min, amp_max = amplitude_range
        masked_elements = set(masked_elements)

        # Generate report
        element_reports = []
        for i, amplitude in enumerate(characteristic):
            if i in masked_elements:
                # Masked elements should be below inactive threshold,
                # otherwise there is something wrong.
                is_masked = True
                state = None
                if amplitude < inactive_threshold:
                    state = VALID
                else:
                    state = TOO_HIGH_AMPLITUDE
            else:
                is_masked = False
                # Determine neighbourhood, necessary to determine element state.
                if group_size == "all":
                    l, r = 0, _N_ELEMENTS
                else:
                    l = i-(math.ceil(group_size/2)-1)
                    l = max(l, 0)
                    r = i+group_size//2+1
                    r = min(r, _N_ELEMENTS)
                near = characteristic[l:r]
                active_elements = np.argwhere(near >= inactive_threshold)
                # TODO what if almost all near elements are inactive?
                # Exclude the current element.
                near = near[active_elements]
                center = np.median(near)
                lower_bound = center*amp_min
                upper_bound = center*amp_max
                assert lower_bound <= upper_bound
                if amplitude > upper_bound:
                    state = TOO_HIGH_AMPLITUDE
                    self.log.warning(f'Element {i} : {state} {round((amplitude - center) * 100 / center, 1)} % center={center}')
                elif amplitude < lower_bound:
                    state = TOO_LOW_AMPLITUDE
                    self.log.warning(f'Element {i} : {state} {round((center - amplitude) * 100 / center, 1)} % center={center}')
                else:
                    state = VALID
            element_reports.append(ProbeElementHealthReport(
                is_masked=is_masked, state=state, amplitude=amplitude))
        return element_reports


def _init_rf_display(width, height):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches((3, 7))
    ax.set_xlabel("channels")
    ax.set_ylabel("Depth (samples)")
    canvas = plt.imshow(np.zeros((height, width)), vmin=-100, vmax=100)
    fig.show()
    return fig, ax, canvas


def _display_rf_frame(frame_number, data, figure, ax, canvas):
    import matplotlib.pyplot as plt
    canvas.set_data(data[frame_number, :, :])
    ax.set_aspect("auto")
    figure.canvas.flush_events()
    plt.draw()


def _init_summary_display():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel("Channel")
    ax.set_ylabel("Amplitude")
    elements = np.arange(_N_ELEMENTS)
    canvas,  = plt.plot(elements, np.zeros((_N_ELEMENTS)))
    ax.set_ylim(bottom=-30000, top=30000)
    fig.show()
    return fig, ax, canvas


def _display_summary(data, figure, ax, canvas):
    import matplotlib.pyplot as plt
    canvas.set_ydata(data)
    ax.set_aspect("auto")
    figure.canvas.flush_events()
    plt.draw()

class StdoutLogger:
    def __init__(self):
        for func in ("debug", "info", "error", "warning"):
            setattr(self, func, self.log)

    def log(self, msg):
        print(msg)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Channels mask test.")
    parser.add_argument("--cfg_path", dest="cfg_path", help="Path to us4r.prototxt configuration file.",
                        required=True)
    parser.add_argument("--threshold", dest="threshold",
                        help="A sample value threshold that determines if the "
                             "given channel is turned off.",
                        required=False, type=float, default=550)
    parser.add_argument("--display_frame", dest="display_frame",
                        help="Select a frame to be displayed. Optional, if not chosen, max amplitude will be displayed.",
                        required=False, type=int, default=None)
    parser.add_argument("--n", dest="n",
                        help="Number of full Lin Sequence TXs to run.",
                        required=False, type=int, default=1)
    parser.add_argument("--method", dest="method",
                        help="Probe element check method.",
                        required=False, type=str, default="neighborhood",
                        choices=["neighborhood", "threshold"])
    parser.add_argument("--rf_file", dest="rf_file",
                        help="The name of the output file with RF data.",
                        required=False, default=None)
    parser.add_argument("--rx_aperture_size", dest="rx_aperture_size",
                        help="The number of receive elements.",
                        required=False, type=int, default=64)

    args = parser.parse_args()

    cfg_path = args.cfg_path

    if args.display_frame is not None:
        fig, ax, canvas = _init_rf_display(_N_ELEMENTS, _N_SAMPLES)
    else:
        fig, ax, canvas = _init_summary_display()
    args = parser.parse_args()

    verifier = ProbeHealthVerifier(log=StdoutLogger())
    if args.method == "neighborhood":
        report = verifier.check_probe_by_neighborhood(cfg_path=cfg_path,
                                                      n=args.n, rx_aperture_size=args.rx_aperture_size)
    elif args.method == "threshold":
        report = verifier.check_probe_by_threshold(cfg_path=cfg_path, n=args.n, rx_aperture_size=args.rx_aperture_size)
    else:
        raise ValueError(f"Unrecognized method: {args.method}")

    if args.display_frame is None:
        _display_summary(report.characteristic, fig, ax, canvas)
        time.sleep(1)

    if args.display_frame is not None:
        for frame in report.data:
            _display_rf_frame(args.display_frame, frame, fig, ax, canvas)
            time.sleep(0.3)

    if args.rf_file is not None:
        pickle.dump(report.data, open(args.rf_file, "wb"))

    # Print information about element health.
    elements_report = report.elements
    invalid_elements = [(i, e) for i, e in enumerate(elements_report) if e.state != "VALID"]
    if len(invalid_elements) == 0:
        print("All channels seems to work correctly.")
    else:
        print("Found invalid channels:")
        for element in invalid_elements:
            print(f"Element {element}")
    print("Close the window to exit")
    plt.show()


