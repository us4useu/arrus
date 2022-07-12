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
python probe_check.py --cfg_path /home/user/us4r.prototxt --method threshold
python probe_check.py --cfg_path /home/user/us4r.prototxt --method neighborhood --rf_file rf.pkl
"""
import time
import os
import collections
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from arrus.utils.probe_check import ProbeHealthVerifier, FeatureDescriptor
from arrus.utils.probe_check import (
    DURATION,
    AMPLITUDE,
    ENERGY
)


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
            rf = data[iframe, k, minsamp:maxsamp, int(nrx/2) - 1]
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


def display_summary(n_elements, report):
    fig, axes = plt.subplots(len(features), 1)
    elements = np.arange(n_elements)
    canvases = []
    for ax, feature in zip(axes, features):
        ax.set_xlabel("Channel")
        ax.set_ylabel(feature.name)
        canvas, = ax.plot(elements, np.zeros((n_elements)))
        ax.set_ylim(bottom=-3000, top=3000)
    fig.show()
    return fig, axes, canvases


def print_health_info(report, detailed=False):
    features = report.params["features"]
    method = report.params["method"]
    elements_report = report.elements
    invalid_elements = []
    invalid_elements_numbers = []
    for i, e in enumerate(elements_report):
        for j, f in enumerate(e.features):
            if f.verdict != "VALID":
                invalid_elements.append(e)
                invalid_elements_numbers.append(e.element_number)

    for feature in features:
        if len(invalid_elements) == 0:
            print(f"Testing {feature.name} by {method} - all channels seems to "
                  f"work correctly.")
        else:
            print(f"Testing {feature.name} by {method} - "
                  f"found {len(invalid_elements)} suspect channels: ")
            print(invalid_elements_numbers)

            active_verdict = collections.defaultdict(list)
            inactive_verdict = collections.defaultdict(list)
            keys = set()
            for element in invalid_elements:
                e_nr = element.element_number
                for feature in element.features:
                    verdict_id = feature.verdict
                    keys.add(verdict_id)
                    if element.active:
                        active_verdict[verdict_id].append(e_nr)
                    else:
                        inactive_verdict[verdict_id].append(e_nr)
            keys = sorted(list(keys))
            for k in keys:
                active_elements = active_verdict[k]
                inactive_elements = inactive_verdict[k]
                if len(active_elements) > 0:
                    print(f"Following active channels have {feature.name} {k}: "
                          f"{active_elements}")
                if len(inactive_elements) > 0:
                    print(f"Following masked channels have {feature.name} {k}: "
                          f"{active_elements}")
            print(" ")

        if detailed:
            for element in invalid_elements:
                active = not element.is_masked
                state = "active" if active else "masked"
                for feature in element.features:
                    print(f"channel# {element.element_number}"
                          f"    state: {state}"
                          f"    feature: {feature.name} \n"
                          f"    verdict: {feature.verdict}\n"
                          f"    value: {np.round(feature.value, 2)}\n"
                          f"    valid range: {element.valid_range}\n"
                          )


def main():
    parser = argparse.ArgumentParser(description="Channels mask test.")
    parser.add_argument("--cfg_path", dest="cfg_path",
                        help="Path to the system configuration file.",
                        required=True)
    parser.add_argument("--display_frame", dest="display_frame",
                        help="Select a frame to be displayed. Optional, if not "
                             "chosen, max amplitude will be displayed.",
                        required=False, type=int, default=None)
    parser.add_argument("--n", dest="n",
                        help="Number of full Lin Sequence TXs to run.",
                        required=False, type=int, default=1)
    parser.add_argument("--method", dest="method",
                        help="Probe element check method.",
                        required=False, type=str, default="threshold",
                        choices=["neighborhood", "threshold"])
    parser.add_argument("--rf_file", dest="rf_file",
                        help="The name of the output file with RF data.",
                        required=False, default=None)
    args = parser.parse_args()

    cfg_path = args.cfg_path
    verifier = ProbeHealthVerifier(log=StdoutLogger())

    features = [
        FeatureDescriptor(
            name=AMPLITUDE,
            active_range=(200, 20000),  # [a.u.]
            masked_elements_range=(0, 2000)  # [a.u.]
        ),
        FeatureDescriptor(
            name=DURATION,
            active_range=(0, 800),  # number of samples
            masked_elements_range=(800, np.inf)  # number of samples
        ),
        FeatureDescriptor(
            name=ENERGY,
            active_range=(0, 15),  # [a.u.]
            masked_elements_range=(0, np.inf)  # [a.u.]
        ),
    ]
    report = verifier.check_probe(cfg_path=cfg_path, n=args.n,
                                  features=features, method=args.method)

    print_health_info(report, detailed=False)

    n_elements = report.sequence_metadata.context.device.probe.model.n_elements
    start_sample, end_sample = report.sequence_metadata.raw_sequence.ops[
        0].rx.sample_range
    n_samples = end_sample - start_sample

    if args.display_frame is not None:
        # Display the sequence of RF frames
        fig, ax, canvas = init_rf_display(n_elements, n_samples)
        for frame in report.data:
            display_rf_frame(args.display_frame, frame, fig, ax, canvas)
            time.sleep(0.3)
        visual_evaluation(report)
        plt.show()
    else:
        display_summary(n_elements, report)
    if args.rf_file is not None:
        pickle.dump(report.data, open(args.rf_file, "wb"))
    print("Close the window to exit")


if __name__ == "__main__":
    main()
