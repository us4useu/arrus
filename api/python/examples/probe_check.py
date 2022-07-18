"""
This is a python script for evaluating transducers in ultrasound probe.
The evaluation is made on the basis of values of features
estimated from signals acquired by each transducer.
The transmit-receive scheme includes transmitting by single transducer
and receiving by transducers in aperture centered on the transmitter.
The features are amplitude, pulse duration and energy of normalised signal.

There are two methods of evaluation.
1. 'Threshold' - bases on the values of features.
If the value is outside predefined range,
the corresponding transducer is treated as 'suspected'.
These ranges can be set at the beginning of the script
(below the line "# features values ranges").
2. 'Neighborhood'  - bases on the values of features
estimated from signals acquired from transducers in close neighborhood
of the examined transducer.
If the value from the examined transducer differs
from the values from other transducers by more then arbitrary percentage,
the transducer is treated as 'suspected'.

At the end of its run, the scripts (optionally) display figures with values of
the features, and initial segments of signals from all transducer, for visual
evaluation.

HOW TO USE THE SCRIPT
The script is called from command line with some options.
Following options are accepted:
--cfg_path: required, path to the system's configuration file,
--help : optional,  displays help,
--method : optional, determines which method will be used ('threshold' (default)
  or 'neighborhood'),
--rf_file : optional, determines the name of possible output file with rf data,
--display_frame : optional, determines if script will display figures,
--n : optional, the number of full Tx cycles to run

Examples:
python probe_check.py --help
python probe_check.py --cfg_path /home/user/us4r.prototxt
python probe_check.py --cfg_path /home/user/us4r.prototxt --rf_file rf.pkl
"""
import collections
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from arrus.utils.probe_check import *


# ------------------------------------------ Utility functions
def visual_evaluation(report, minsamp=0, maxsamp=512, nx=16, figsize=(16, 8),
                      iframe=0):
    """
    The function plot selected fragments of all signals in input array on a
    single figure.
    It is assumed that signal array shape is number of transducers (rows)
    x number of samples (columns).
    """
    import matplotlib.lines as mlines
    data = report.data
    nframe, ntx, nsamp, nrx = data.shape

    ny = int(np.ceil(ntx / nx)) + 1  # + 1 row for the legend
    fig, ax = plt.subplots(ny, nx, figsize=figsize)
    k = 0
    status_color = ["C1" if e.is_masked else "C0" for e in report.elements]

    for j in range(nx):
        ax[0, j].axis("off")

    for i in range(1, ny):
        for j in range(nx):
            rf = data[iframe, k, minsamp:maxsamp, int(nrx / 2) - 1]
            rf = rf - np.mean(rf)
            ax[i, j].plot(rf, c=status_color[k])
            ax[i, j].set_title(f"{k}")
            ax[i, j].axis("off")
            k += 1
    lact = mlines.Line2D([], [], color="C0", label="active")
    lina = mlines.Line2D([], [], color="C1", label="inactive")
    fig.legend(handles=[lact, lina])
    fig.tight_layout()


def init_rf_display(width, height):
    fig, ax = plt.subplots()
    fig.set_size_inches((3, 7))
    ax.set_xlabel("channels")
    ax.set_ylabel("Depth (samples)")
    canvas = plt.imshow(np.zeros((height, width)), vmin=-100, vmax=100)
    fig.show()
    return fig, ax, canvas


def display_rf_frame(frame_number, data, figure, ax, canvas):
    canvas.set_data(data[frame_number, :, :])
    ax.set_aspect("auto")
    figure.canvas.flush_events()
    plt.draw()


def display_summary(n_elements: int, report: ProbeHealthReport):
    characteristics = report.characteristics
    fig, axes = plt.subplots(len(characteristics), 1)
    elements = np.arange(n_elements)
    canvases = []
    for i, (name, c) in enumerate(characteristics.items()):
        ax = axes[i]
        ax.set_xlabel("Channel")
        ax.set_ylabel(name)
        ax.plot(elements, c)
    fig.show()
    return fig, axes


def print_health_info(report):
    features = report.params["features"]
    elements_report = report.elements

    invalid_els = collections.defaultdict(list)
    # tuples: (nr, is_masked, element_feature_descriptor)

    for i, e in enumerate(elements_report):
        for name, f in e.features.items():
            if f.verdict != ElementValidationVerdict.VALID:
                invalid_els[name].append((e.element_number, e.is_masked, f))

    for feature in features:
        print(f"Test results for feature: {feature.name}")
        feature_invalid_elements = invalid_els[feature.name]
        nrs, _, _ = zip(*feature_invalid_elements)
        if len(nrs) == 0:
            print("All channels seems to work correctly.")
        else:
            print(f"Found {len(feature_invalid_elements)} suspect channels: ")
            print(nrs)

            for nr, is_masked, f_el_desc in invalid_els:
                state = "masked" if is_masked else "active"
                result = f_el_desc.validation_result
                print(f"channel# {nr}, state: {state}, "
                      f"verdict: {result.verdict}, "
                      f"value: {np.round(f_el_desc.value, 2)}, "
                      f"valid range: {result.valid_range}.")


def get_data_dimensions(metadata):
    n_elements = metadata.context.device.probe.model.n_elements
    sequence = metadata.raw_sequence
    start_sample, end_sample = sequence.ops[0].rx.sample_range
    n_samples = end_sample - start_sample
    return n_elements, n_samples


def main():
    parser = argparse.ArgumentParser(description="Channels mask test.")
    parser.add_argument("--cfg_path", dest="cfg_path",
                        help="Path to the system configuration file.",
                        required=True)
    parser.add_argument("--display_frame", dest="display_frame",
                        help="Select a frame to be displayed. Optional, if not "
                             "chosen, summary of features will be displayed.",
                        required=False, type=int, default=None)
    parser.add_argument("--n", dest="n",
                        help="Number of full TX/RX sequences to run.",
                        required=False, type=int, default=1)
    parser.add_argument("--rf_file", dest="rf_file",
                        help="The name of the output file with RF data.",
                        required=False, default=None)
    args = parser.parse_args()

    cfg_path = args.cfg_path
    verifier = ProbeHealthVerifier()

    features = [
        FeatureDescriptor(
            name=MaxAmplitudeExtractor.feature,
            active_range=(200, 20000),  # [a.u.]
            masked_elements_range=(0, 2000)  # [a.u.]
        ),
        FeatureDescriptor(
            name=SignalDurationTimeExtractor.feature,
            active_range=(0, 800),  # number of samples
            masked_elements_range=(800, np.inf)  # number of samples
        ),
        FeatureDescriptor(
            name=EnergyExtractor.feature,
            active_range=(0, 15),  # [a.u.]
            masked_elements_range=(0, np.inf)  # [a.u.]
        ),
    ]
    validator = ByNeighborhoodValidator()
    report = verifier.check_probe(cfg_path=cfg_path, n=args.n,
                                  features=features, validator=validator)

    print_health_info(report)
    n_elements, n_samples = get_data_dimensions(report.sequence_metadata)

    if args.display_frame is not None:
        # Display the sequence of RF frames
        fig, ax, canvas = init_rf_display(n_elements, n_samples)
        for frame in report.data:
            display_rf_frame(args.display_frame, frame, fig, ax, canvas)
            time.sleep(0.3)
        plt.show()
        #  Display all waveforms in a single window.
        visual_evaluation(report)
    else:
        display_summary(n_elements, report)
    if args.rf_file is not None:
        pickle.dump(report.data, open(args.rf_file, "wb"))
    print("Close the window to exit")


if __name__ == "__main__":
    main()
