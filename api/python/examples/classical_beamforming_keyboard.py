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
import copy
import keyboard

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
    # TODO use the imaging pipeline
    print(f"Displaying frame {frame_number}")
    bmode, metadata = imaging_pipeline(cp.asarray(data), metadata)
    canvas.set_data(bmode)
    ax.set_aspect("auto")
    figure.canvas.flush_events()
    plt.draw()


rf_data = []
rf_metadata = []


def save_raw_data(frame_number, data, metadata):
    print(f"Data shape: {data.shape}")
    arrus.logging.log(arrus.logging.INFO, f"Saving frame {frame_number}")
    np.save(f"rf_{frame_number}.npy", data)
    with open(f"metadata_{frame_number}.pkl", "wb") as file:
        pickle.dump(metadata, file)


def copy_raw_data(frame_number, data, metadata):
    global rf_data, rf_metadata
    rf_data.append(data.copy())
    frame_metadata = metadata.custom["frame_metadata_view"].copy()
    custom_data = copy.copy(metadata.custom)
    custom_data["frame_metadata_view"] = frame_metadata
    metadata = metadata.copy(custom=custom_data)
    rf_metadata.append(metadata)


def create_bmode_imaging_pipeline(decimation_factor=4, cic_order=2,
                                  x_grid=None, z_grid=None):
    if x_grid is None:
        x_grid = np.arange(-50, 50, 0.4)*1e-3
    if z_grid is None:
        z_grid = np.arange(0, 60, 0.4)*1e-3

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
            ToGrayscaleImg()))


def get_rf_iq_data(buffer, buffer_size):
    iq_rec = iq_reconstruct(decimation_factor=4, cic_order=2)
    iq_data_list = []
    for i in range(buffer_size):
        data, metadata = buffer.tail()
        iq_data, iq_metadata = iq_rec(cp.asarray(data), metadata)
        iq_data_list.append((iq_data.get(), iq_metadata.get()))
        buffer.release_tail()
    return iq_data_list


def iq_reconstruct(decimation_factor=4, cic_order=2):
    return Pipeline(
        steps=(
            RemapToLogicalOrder(),
            Transpose(axes=(0, 2, 1)),
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=decimation_factor,
                       cic_order=cic_order),
            RxBeamforming()))


iq_data = []
iq_metadata = []

current_voltage = 30

def save_iq_data(frame_number, data, metadata, iq_rec):
    global iq_data, iq_metadata
    iq, iq_m = iq_rec(cp.asarray(data), metadata)

    iq_data.append(iq.get())
    frame_metadata = metadata.custom["frame_metadata_view"].copy()
    custom_data = copy.copy(metadata.custom)
    custom_data["frame_metadata_view"] = frame_metadata
    metadata = metadata.copy(custom=custom_data)
    iq_metadata.append(metadata)


def main():
    parser = argparse.ArgumentParser(
        description="The script acquires a sequence of RF data or "
                    "reconstructs b-mode images.")

    parser.add_argument("--cfg", dest="cfg",
                        help="Path to session configuration file.",
                        required=True)
    parser.add_argument("--action", dest="action",
                        help="An action to perform.",
                        required=True, choices=["nop", "save", "img", "save_mem", "save_iq"])
    parser.add_argument("--n", dest="n",
                        help="How many times should the operation be performed.",
                        required=False, type=int, default=100)
    parser.add_argument("--host_buffer_size", dest="host_buffer_size",
                        help="Host buffer size.", required=False, type=int,
                        default=2)
    parser.add_argument("--rx_batch_size", dest="rx_batch_size",
                        help="Rx batch size.", required=False, type=int,
                        default=1)
    args = parser.parse_args()

    x_grid = np.arange(-50, 50, 0.4)*1e-3
    z_grid = np.arange(0, 60, 0.4)*1e-3

    seq = LinSequence(
        tx_aperture_center_element=np.arange(8, 183),
        tx_aperture_size=64,
        tx_focus=30e-3,
        pulse=Pulse(center_frequency=8e6, n_periods=3.5, inverse=False),
        rx_aperture_center_element=np.arange(8, 183),
        rx_aperture_size=64,
        rx_sample_range=(0, 2048),
        pri=100e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1490)

    bmode_imaging = create_bmode_imaging_pipeline(x_grid=x_grid, z_grid=z_grid)
    iq_rec = iq_reconstruct(4, 2)

    if args.action == "img":
        fig, ax, canvas = init_display(len(z_grid), len(x_grid))

    action_func = {
        "nop":  None,
        "save": save_raw_data,
        "save_mem": copy_raw_data,
        "img":  lambda frame_number, data, metadata: display_data(
            frame_number, data, metadata, bmode_imaging, fig, ax, canvas),
        "save_iq": lambda frame_number, data, metadata: save_iq_data(
            frame_number, data, metadata, iq_rec),
    }[args.action]

    # Here starts communication with the device.
    session = arrus.session.Session(args.cfg)

    us4r = session.get_device("/Us4R:0")
    gpu = session.get_device("/GPU:0")

    # Set the pipeline to be executed on the GPU
    bmode_imaging.set_placement(gpu)
    iq_rec.set_placement(gpu)

    # Set initial voltage on the us4r-lite device.
    # Upload sequence on the us4r-lite device.
    buffer = us4r.upload(seq, mode="sync",
                         host_buffer_size=args.host_buffer_size,
                         rx_batch_size=args.rx_batch_size)

    def increase_voltage(ev):
        print("Increasing voltage")
        global current_voltage
        if current_voltage >= 90:
            print("maximum voltage set")
            return
        current_voltage += 1
        us4r.set_hv_voltage(current_voltage)

    def decrease_voltage(ev):
        print("Decreasing voltage")
        global current_voltage
        if current_voltage <= 0:
            print("minimum voltage set")
        current_voltage -= 1
        us4r.set_hv_voltage(current_voltage)

    keyboard.on_press_key("a", increase_voltage)
    keyboard.on_press_key("1", decrease_voltage)

    # Start the device.
    us4r.start()
    times = []
    arrus.logging.log(arrus.logging.INFO, f"Running {args.n} iterations.")
    for i in range(args.n):
        start = time.time()
        data, metadata = buffer.head()

        if action_func is not None:
            action_func(i, data, metadata)

        buffer.release_tail()
        times.append(time.time()-start)
    arrus.logging.log(arrus.logging.INFO,
         f"Done, average acquisition + processing time: {np.mean(times)} [s]")

    # rf_iq_data_buffer = get_rf_iq_data(buffer, 100)
    if args.action == "save_mem":
        print("Saving data to rf.npy i metadata.pkl")
        global rf_data, rf_metadata
        np.save("rf.npy", np.stack(rf_data))
        with open("metadata.pkl", "wb") as f:
            pickle.dump(rf_metadata, f)
    if args.action == "save_iq":
        global iq_data, iq_metadata
        np.save("rf_iq.npy", np.stack(iq_data))
        with open("metadata.pkl", "wb") as f:
            pickle.dump(iq_metadata, f)

    print("Stopping the device.")
    us4r.stop()
    print("Device stopped.")


if __name__ == "__main__":
    main()
