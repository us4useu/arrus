"""
This is a python script for evaluating transducers in ultrasound probe.
The evaluation is made on the basis of values of features
estimated from signals acquired by each transducer.
The probe should be 'in the air' - the analysed signal is assumed
to be reflection from the lens surface.
The transmit-receive scheme includes transmitting by single transducer
and receiving by transducers in aperture centered on the transmitter.

Current features are
1. amplitude,
2. pulse duration,
3. energy of normalised signal,
4. pearson correlation coefficient (PCC) with 'footprint' signal.
PCC require a footprint (i.e. object containing reference rf signal array).
The footprint should be acquired earlier using the same setup.
This feature measure how much the signals changed comparing to footprint.
When transducer is broken, the acquired signal should be different comparing
to signal acquired before damage.

There are two methods of evaluation.
1. 'Threshold'
It bases on the values of features estimated
from signals acquired from examined transducer.
If the value is outside predefined range,
the corresponding transducer is treated as 'suspected'.
These ranges can be set in FeatureDescriptor() calls in the main() function the script.
(below "# define features list").
This ranges can differs for different probes, and should be set by user.
2. 'Neighborhood'
It bases on the values of features estimated
from signals acquired from examined transducer and transducers
in close neighborhood.
If the value from the examined transducer differs
from the values from other transducers by more then arbitrary percentage,
the transducer is treated as 'suspected'.
Note, that PCC should be validated using threshold method
(i.e. usign ByThresholdValidator()).

At the end of its run, the scripts (optionally) display figures with values of
the features, and initial segments of signals from all transducer, for visual
evaluation.

HOW TO USE THE SCRIPT
The script is called from command line with some options.
Following options are accepted:
(required)
--cfg_path: path to the system's configuration file,

(optional)
--help : displays help,
--method : determines which method will be used ('threshold' (default)
  or 'neighborhood'),
--rf_file : determines the name of optional output file with rf data,
--create_footprint : creates footprint and store it in given file,
--use_footprint : determines which footprint file to use,
--n : the number of full Tx cycles to run (default 8),
--tx_frequency : determines transmit frequency in [Hz] (default 8e6),
--nrx : determines the size of receiving aperture (default 32),
--display_tx_channel : displays signals received by transmision of a given channel,
--show_pulse_comparison : displays acquired pulse and corresponding
    footprint signal,
--features: features to evaluation (amplitude, energy, duration, pcc)
--display_summary : (flag) displays features values on figure,
--visual_eval : (flag) displays pulses from all channels in single figure,

Examples:
python check_probe.py --help
python check_probe.py --cfg_path /home/user/us4r.prototxt
python check_probe.py --cfg_path /home/user/us4r.prototxt --rf_file rf.pkl
python check_probe.py --cfg_path ~/us4r.prototxt --create_footprint footprint.pkl --n=16
python check_probe.py --cfg_path ~/us4r.prototxt --use_footprint footprint.pkl
python check_probe.py --cfg_path ~/us4r.prototxt --use_footprint footprint.pkl --features amplitude pcc
python check_probe.py --cfg_path ~/us4r.prototxt --display_tx_channel 64
python check_probe.py --cfg_path ~/us4r.prototxt --display_summary

Additional notes:
1. This script tries to identify channels it considers suspicious.
In the case of a suspicious channel, the user should verify it manually
(i.e. to look on the signal from the channel and decide, if it is good or not),
because the features used are only indicative.
2. Some features (like pulse duration) are more 'sensitive' than the others,
i.e. are more likely to give false positives.
If there are a lot of false positives one can remove the feature from the list
of features to verify.
3. Using of threshold method assume that one know the range of values
of verified features.
These ranges often are different for different probes.
If one do not know the feature range of values, the neighborhood method can be used.
4. The use of probe footprint seems to be the most convenient, however it require
to create a footprint (using --create_footprint option).
It assume, that all channels are ok during footprint creation, and later it
just indentify channels where signals not match with the footprint.
Thus, when some channel is damaged during footprint creation, it may be undetected
on later tests.
5. Remember to set proper tx_frequency (using --tx_frequency) when create a footprint
or making tests without it.
When footprint is used, tx/rx scheme parameters will be get from it.

"""


import collections
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from arrus.utils.probe_check import *

# ------------------------- Utility functions ---------------------------------

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
    fig.show()
    plt.show()


def init_rf_display(width, height):
    fig, ax = plt.subplots()
    fig.set_size_inches((3, 7))
    ax.set_xlabel("channels")
    ax.set_ylabel("Depth (samples)")
    canvas = plt.imshow(np.zeros((height, width)), vmin=-100, vmax=100)
    fig.show()
    return fig, ax, canvas


def display_rf(rf, figure, ax, canvas):
    canvas.set_data(rf)
    ax.set_aspect("auto")
    # figure.canvas.flush_events()
    # plt.draw()


def display_summary(n_elements: int, report: ProbeHealthReport):
    characteristics = report.characteristics
    fig, axes = plt.subplots(len(characteristics), 1, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    elements = np.arange(n_elements)
    canvases = []
    for i, (name, c) in enumerate(characteristics.items()):
        ax = axes[i]
        ax.set_ylabel(name)
        ax.plot(elements, c)
    ax.set_xlabel("Channels")
    fig.show()
    plt.show()
    return fig, axes


def print_health_info(report):
    features = report.params["features"]
    elements_report = report.elements

    invalid_els = collections.defaultdict(list)
    # tuples: (nr, is_masked, element_feature_descriptor)

    for i, e in enumerate(elements_report):
        for name, f in e.features.items():
            if f.validation_result.verdict != ElementValidationVerdict.VALID:
                invalid_els[name].append((e.element_number, e.is_masked, f))

    for feature in features:
        print("----------------------------------------------")
        print("Test results:")
        print(f"  feature: {feature.name}")
        print(f"   method: {report.validator.name}")
        feature_invalid_elements = invalid_els[feature.name]

        if len(feature_invalid_elements) == 0:
            nrs = []
        else:
            nrs, _, _ = zip(*feature_invalid_elements)
        if len(nrs) == 0:
            print("All channels seems to work correctly.")
        else:
            print(f"Found {len(feature_invalid_elements)} suspect channels: ")
            for nr, is_masked, f_el_desc in feature_invalid_elements:
                state = "masked" if is_masked else "active"
                result = f_el_desc.validation_result
                print(f"channel# {nr}, state: {state}, "
                      f"verdict: {result.verdict}, "
                      f"value: {np.round(f_el_desc.value, 2)}, "
                      f"valid range: {result.valid_range}.")


def get_data_dimensions(metadata, probe_nr):
    n_elements = metadata.context.device.probe[probe_nr].model.n_elements
    sequence = metadata.context.raw_sequence
    start_sample, end_sample = sequence.ops[0].rx.sample_range
    n_samples = end_sample - start_sample
    return n_elements, n_samples


def load_footprint(path):
    if isinstance(path, str) is False:
        return None
    else:
        with open(path, "rb") as f:
            print("Loading footprint data...")
            data = pickle.load(f)
            print("Footprint data loaded.")
        return data


def show_footprint_pulse_comparison(
        footprint,
        report,
        itx,
        iframe=0,
        smp=None,
        irx=None,
):
    """
    Show plot of given signal and corresponding footprint signal.

    :param footprint: Footprint object
    :param report: ProbeHealthReport object (contains rf signal array)
    :param itx: int, channel number
    :iframe: int, frame number (optional - default 0)
    :smp: slice, samples range (optional)
    :irx: receiving aperture channel number
         (optional - default correponds with itx)
    """
    rf = report.data
    if footprint is not None and rf.shape != footprint.rf.shape:
        raise ValueError(
            "The input rf array has different shape than footprint.rf")
    _, _,nsmp, nrx = rf.shape
    if smp is None:
        smp = slice(0, nsmp)
    if irx is None:
        irx = int(np.ceil(nrx / 2) - 1)
    plt.plot(rf[iframe, itx, smp, irx])
    if footprint is not None:
        plt.plot(footprint.rf[iframe, itx, smp, irx])
    plt.legend(["rf", "footprint rf"])
    plt.xlabel("samples")
    plt.ylabel("[a.u.]")
    plt.title(f"channel {itx}")
    plt.show()

def main():

    # parse input parameters
    parser = argparse.ArgumentParser(description="Channels mask test.")
    parser.add_argument(
        "--cfg_path", dest="cfg_path",
        help="Path to the system configuration file.",
        required=True,
    )
    parser.add_argument(
        "--display_tx_channel", dest="display_tx_channel",
        help="Display received signals after transmitting by given channel.",
        required=False,
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n", dest="n",
        help="Number of full TX/RX sequences to run.",
        required=False,
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tx_frequency", dest="tx_frequency",
        help="Pulse transmit frequency.",
        required=False,
        type=float,
        default=8e6,
    )
    parser.add_argument(
        "--nrx", dest="nrx",
        help="Size of receiving aperture",
        required=False,
        type=int,
        default=32,
    )
    parser.add_argument(
        "--rf_file", dest="rf_file",
        help="The name of the output file with RF data.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--use_footprint", dest="footprint",
        help="The name of the footprint file with RF data "
             "to be loaded for comparison.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--create_footprint", dest="create_footprint",
        help="The name of the footprint file with RF data "
             "to be created .",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--show_pulse_comparison", dest="show_pulse_comparison",
        help="Shows pulses (acquired and footprint) from selected channel.",
        required=False,
        type=int,
        default=None,
    )
    parser.add_argument(
        "--method", dest="method",
        help="Method used - threshold (default) or neighborhood",
        required=False,
        default="threshold",
    )
    parser.add_argument(
        "--features", dest="features",
        help="List of features used for evaluation",
        required=False,
        default="all",
        nargs="+",
    )
    parser.add_argument(
        "--display_summary", dest="display_summary",
        help="Display features values in all channels.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--visual_eval", dest="visual_eval",
        help="Show pulses from all channels on single figure.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--probe_nr", dest="probe_nr",
        help="Number of the probe to use.",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hpf_corner_frequency", dest="hpf_corner_frequency",
        help="AFE High-pass filter corner frequency to apply, None means that the system's default value should be used.",
        required=False,
        type=int,
        default=None,
    )


    args = parser.parse_args()

    verifier = ProbeHealthVerifier()

    # create footprint and quit (when --create_footprint option is used)
    if args.create_footprint is not None:
        footprint = verifier.get_footprint(
            cfg_path=args.cfg_path,
            n=args.n,
            tx_frequency=args.tx_frequency,
            nrx=args.nrx,
        )
        pickle.dump(footprint, open(args.create_footprint, "wb"))
        print("---------------------------------------------------------------")
        print("The footptint have been created "
              f"and store in {args.create_footprint} file.")
        print(f"The script {__file__} ends here.")
        print("---------------------------------------------------------------")
        quit()

    footprint = load_footprint(args.footprint)

    # create validator
    if args.method == "threshold":
        validator = ByThresholdValidator()
    elif args.method == "neighborhood":
        validator = ByNeighborhoodValidator()
    else:
        import warnings
        warnings.warn(
            "Unknown method - using default (threshold)."
        )
        validator = ByThresholdValidator()

    # prepare examined features list
    available_features = [
        "amplitude",
        "duration",
        "energy",
        "pcc"
    ]
    can_use_footprint = args.method == "threshold" and footprint is not None
    if args.features == 'all':
        given_features = available_features
        if not can_use_footprint:
            given_features.pop(3)
    else:
        given_features = args.features
    features = []
    for feat in given_features:
        if feat == "amplitude":
            features.append(
                FeatureDescriptor(
                    name=MaxAmplitudeExtractor.feature,
                    active_range=(0, 3000),  # [a.u.]
                    masked_elements_range=(0, 3000)  # [a.u.]
                )
            )
        elif feat == "duration":
            features.append(
                FeatureDescriptor(
                    name=SignalDurationTimeExtractor.feature,
                    active_range=(0, 1000),  # number of samples
                    masked_elements_range=(200, np.inf)  # number of samples
                )
            )
        elif feat == "energy":
            features.append(
                FeatureDescriptor(
                    name=EnergyExtractor.feature,
                    active_range=(0, 200),  # [a.u.]
                    masked_elements_range=(0, np.inf)  # [a.u.]
                )
            )
        elif feat == "pcc":
            if can_use_footprint:
                features.append(
                    FeatureDescriptor(
                        name=FootprintSimilarityPCCExtractor.feature,
                        active_range=(0.5, 1),  # [a.u.]
                        masked_elements_range=(0, 1)  # [a.u.]
                    )
                )
        else:
            raise ValueError(f"{feat} is a bad feature name")

    # check probe
    report = verifier.check_probe(
        cfg_path=args.cfg_path,
        n=args.n,
        tx_frequency=args.tx_frequency,
        nrx=args.nrx,
        features=features,
        validator=validator,
        footprint=footprint,
        probe_nr=args.probe_nr,
        hpf_corner_frequency=args.hpf_corner_frequency
    )

    # show results
    print_health_info(report)
    print("----------------------------------------------")

    n_elements, n_samples = get_data_dimensions(report.sequence_metadata, probe_nr=args.probe_nr)

    if args.display_tx_channel is not None:
        fig, ax, canvas = init_rf_display(n_elements, n_samples)
        ax.set_title(f"tx channel: {args.display_tx_channel}")
        display_rf(
            report.data[0, args.display_tx_channel, :, :],
            fig,
            ax,
            canvas
        )
        plt.show()

    #  Display all waveforms in a single window.
    if args.visual_eval:
        visual_evaluation(report)

    if args.display_summary:
        display_summary(n_elements, report)

    if args.rf_file is not None:
        data = {"rf": report.data, "context": report.sequence_metadata}
        pickle.dump(data, open(args.rf_file, "wb"))

    if args.show_pulse_comparison is not None:
        show_footprint_pulse_comparison(
            footprint,
            report,
            itx=args.show_pulse_comparison,
        )


if __name__ == "__main__":
    main()
