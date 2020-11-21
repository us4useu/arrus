import numpy as np
import argparse
import matplotlib.pyplot as plt
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse
import arrus.logging
import arrus.utils.us4r
import time
import pickle
import dataclasses
import cupy as cp

from arrus.utils.imaging import (
    Pipeline,
    BandpassFilter,
    QuadratureDemodulation,
    Decimation,
    RxBeamforming,
    EnvelopeDetection,
    Transpose,
    ScanConversion,
    LogCompression,
    DynamicRangeAdjustment,
    ToGrayscaleImg
)

from arrus.utils.us4r import RemapToLogicalOrder

arrus.set_clog_level(arrus.logging.TRACE)
arrus.add_log_file("test.log", arrus.logging.TRACE)

collected_bmode_data = []
collected_iq_data = []
collected_metadata = []


def init_display(aperture_size, n_samples):
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.set_xlabel("OX")
    ax.set_ylabel("OZ")
    image_w, image_h = aperture_size, n_samples
    canvas = plt.imshow(np.zeros((image_w, image_h)),
                        vmin=np.iinfo(np.uint8).min,
                        vmax=np.iinfo(np.uint8).max,
                        cmap="gray")
    fig.show()
    return fig, ax, canvas


def display_data(frame_number, data, metadata, imaging_pipeline, figure, ax, canvas):
    print(f"Displaying frame {frame_number}")
    bmode, metadata = imaging_pipeline(cp.asarray(data), metadata)
    canvas.set_data(bmode)
    ax.set_aspect("auto")
    figure.canvas.flush_events()
    plt.draw()


def display_data_sequence(seq):
    fig, ax, canvas = init_display(64, 512)
    for data in seq:
        canvas.set_data(data)
        ax.set_aspect("auto")
        fig.canvas.flush_events()
        plt.draw()


def collect_iq_data(data, metadata, iq_pipeline):
    iq_rf, metadata = iq_pipeline(cp.asarray(data), metadata)
    collected_iq_data.append(iq_rf.get())
    collected_metadata.append(metadata)


def collect_bmode_data(data, metadata, bmode_pipeline):
    bmode, metadata = bmode_pipeline(cp.asarray(data), metadata)
    collected_bmode_data.append(bmode.get())
    collected_metadata.append(metadata)


def get_timestamps(metadata):
    timestamps = []
    for m in metadata:
        timestamps.append(m.custom["frame_metadata_view"].copy().view(np.int8)[0:8].view(np.uint64).item())
    return timestamps


def get_average_timestamp_diff(metadata):
    timestamps = get_timestamps(metadata)
    diffs = np.diff(timestamps)
    return np.average(diffs)


def save_raw_data(frame_number, data, metadata):
    arrus.logging.log(arrus.logging.INFO, f"Saving frame {frame_number}")
    np.save(f"rf_{frame_number}.npy", data)
    with open(f"metadata_{frame_number}.pkl", "wb") as file:
        pickle.dump(metadata, file)


def create_bmode_imaging_pipeline(decimation_factor=4, cic_order=2,
                                  x_grid=None, z_grid=None):
    if x_grid is None:
        x_grid = np.arange(-50, 50, 0.2)*1e-3
    if z_grid is None:
        z_grid = np.arange(0, 60, 0.2)*1e-3

    return Pipeline(
        steps=(
            RemapToLogicalOrder(),
            Transpose(axes=(0, 2, 1)),
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=decimation_factor,
                       cic_order=cic_order),
            RxBeamforming(),
            EnvelopeDetection(),
            Transpose(),
            ScanConversion(x_grid=x_grid, z_grid=z_grid),
            LogCompression(),
            DynamicRangeAdjustment(min=5, max=120),
            ToGrayscaleImg()
        )
    )


def create_rf_iq_pipeline(decimation_factor=4, cic_order=2):
    return Pipeline(
        steps=(
            RemapToLogicalOrder(),
            Transpose(axes=(0, 2, 1)),
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=decimation_factor,
                       cic_order=cic_order),
            RxBeamforming()
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="The script acquires a sequence of RF data or "
                    "reconstructs b-mode images.")

    parser.add_argument("--cfg", dest="cfg",
                        help="Path to session configuration file.",
                        required=True)
    parser.add_argument("--action", dest="action",
                        help="An action to perform.",
                        required=True, choices=["nop", "save_rf", "save_iq",
                                                "save_bmode", "img"])
    parser.add_argument("--n", dest="n",
                        help="How many times should the operation be performed.",
                        required=False, type=int, default=100)
    parser.add_argument("--host_buffer_size", dest="host_buffer_size",
                        help="Host buffer size.", required=False, type=int,
                        default=2)

    args = parser.parse_args()

    x_grid = np.arange(-50, 50, 0.4)*1e-3
    z_grid = np.arange(0, 60, 0.4)*1e-3

    seq = LinSequence(
        tx_aperture_center_element=np.arange(8, 183),
        tx_aperture_size=64,
        tx_focus=30e-3,
        pulse=Pulse(center_frequency=5e6, n_periods=3.5, inverse=False),
        rx_aperture_center_element=np.arange(8, 183),
        rx_aperture_size=64,
        rx_sample_range=(0, 2048),
        pri=100e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1490)

    bmode_imaging = create_bmode_imaging_pipeline(x_grid=x_grid, z_grid=z_grid)
    iq_data_pipeline = create_rf_iq_pipeline()

    if args.action == "img":
        fig, ax, canvas = init_display(len(z_grid), len(x_grid))

    action_func = {
        "nop":  None,
        "save_rf": save_raw_data,
        "save_iq": lambda frame_number, data, metadata: collect_iq_data(data, metadata, iq_data_pipeline),
        "save_bmode": lambda data, metadata: collect_bmode_data(data, metadata, bmode_imaging),
        "img":  lambda frame_number, data, metadata: display_data(frame_number, data, metadata, bmode_imaging, fig, ax, canvas)
    }[args.action]

    # Here starts communication with the device.
    session = arrus.session.Session(args.cfg)

    us4r = session.get_device("/Us4R:0")
    gpu = session.get_device("/GPU:0")

    # Set the pipeline to be executed on the GPU
    bmode_imaging.set_placement(gpu)
    iq_data_pipeline.set_placement(gpu)
    # Set initial voltage on the us4r-lite device.
    us4r.set_hv_voltage(30)
    # Upload sequence on the us4r-lite device.
    buffer = us4r.upload(seq, mode="sync",
                         host_buffer_size=args.host_buffer_size)

    # Start the device.
    us4r.start()
    times = []
    arrus.logging.log(arrus.logging.INFO, f"Running {args.n} iterations.")
    for i in range(args.n):
        start = time.time()
        data, metadata = buffer.tail()

        if action_func is not None:
            action_func(i, data, metadata)

        input("Waiting for the input.")

        buffer.release_tail()
        times.append(time.time()-start)
    arrus.logging.log(arrus.logging.INFO,
         f"Done, average acquisition + processing time: {np.mean(times)} [s]")
    us4r.stop()

    global collected_iq_data, collected_bmode_data, collected_metadata
    if args.action == "save_iq":
        collected_iq_data = np.stack(collected_iq_data)
        np.save("iq_data.npy", collected_iq_data)
    if args.action == "save_bmode":
        collected_bmode_data = np.stack(collected_bmode_data)
        np.save("bmode_data.npy", collected_bmode_data)
    if args.action in {"save_iq", "save_bmode"}:
        with open(f"metadata.pkl", "wb") as file:
            pickle.dump(collected_metadata, file)



if __name__ == "__main__":
    main()
