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
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from arrus.utils.probe_check import ProbeHealthVerifier

# ------------------------------------------ Utility functions
class StdoutLogger:
    def __init__(self):
        for func in ("debug", "info", "error", "warning"):
            setattr(self, func, self.log)

    def log(self, msg):
        print(msg)


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


def init_summary_display():
    fig, ax = plt.subplots()
    ax.set_xlabel("Channel")
    ax.set_ylabel("Amplitude")
    elements = np.arange(_N_ELEMENTS)
    canvas, = plt.plot(elements, np.zeros((_N_ELEMENTS)))
    ax.set_ylim(bottom=-3000, top=3000)
    fig.show()
    return fig, ax, canvas


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
    reports = verifier.check_probe(cfg_path=cfg_path,
                                   n=args.n,
                                   features="all",
                                   method=args.method)


    features = []
    print("Following features will be tested:")
    for feature in EXTRACTORS:
        print(feature)
        if feature == DURATION:
            name = DURATION
            active_range = DURATION_ACTIVE_RANGE
            inactive_range = DURATION_INACTIVE_RANGE
        elif feature == AMPLITUDE:
            name = AMPLITUDE
            active_range = AMPLITUDE_ACTIVE_RANGE
            inactive_range = AMPLITUDE_INACTIVE_RANGE
        elif feature == ENERGY:
            name = ENERGY
            active_range = ENERGY_ACTIVE_RANGE
            inactive_range = ENERGY_INACTIVE_RANGE
        else:
            raise ValueError("Bad feature.")
        features.append(
            FeatureDescriptor(
                name=name,
                active_range=active_range,
                inactive_range=inactive_range
            )
        )

    # Present results

    figs = []
    axes = []
    canvases = []
    if args.display_frame is not None:
        _init_fun = lambda: init_rf_display(_N_ELEMENTS, _N_SAMPLES)
    else:
        _init_fun = lambda: init_summary_display()

    for i in range(len(EXTRACTORS)):
        fig, ax, canvas = _init_fun()
        figs.append(fig)
        axes.append(ax)
        canvases.append(canvas)

    for i, report in enumerate(reports):
        # Show visual report
        fig = figs[i]
        ax = axes[i]
        canvas = canvases[i]
        _update_display(report, ax)
        if report.params["method"] == "neighborhood":
            values = report.feature_values_in_neighborhood
        else:
            values = report.feature_values
        values = report.feature_values
        if args.display_frame is None:
            _display_summary(values, fig, ax, canvas)
            time.sleep(1)
        if args.display_frame is not None:
            for frame in report.data:
                display_rf_frame(args.display_frame, frame, fig, ax, canvas)
                time.sleep(0.3)
        # Save report
        if args.rf_file is not None:
            pickle.dump(report.data, open(args.rf_file, "wb"))
        # Print information about element health.
        _print_health_info(report, detailed=False)
        print("Close the window to exit")

    if args.display_frame is not None:
        _visual_evaluation(report)
        plt.show()

if __name__ == "__main__":
    main()
