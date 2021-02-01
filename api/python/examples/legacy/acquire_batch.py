import arrus
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import pickle
import time
import argparse
import os
from datetime import datetime

from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse

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
from arrus.utils.us4r import (
    RemapToLogicalOrder,
    get_batch_data,
    get_batch_metadata
)

# Acquisition parameters.
VOLTAGE = 50
TX_RX_SEQUENCE = LinSequence(
    tx_aperture_center_element=np.arange(8, 183),
    tx_aperture_size=64,
    tx_focus=20e-3,
    pulse=Pulse(center_frequency=8e6, n_periods=3.5, inverse=False),
    rx_aperture_center_element=np.arange(8, 183),
    rx_aperture_size=64,
    rx_sample_range=(0, 2048),
    pri=100e-6,
    tgc_start=14,
    tgc_slope=2e2,
    downsampling_factor=2,
    speed_of_sound=1480)


arrus.set_clog_level(arrus.logging.INFO)
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


prev_timestamp = 0


def display_data(frame_number, data, metadata, imaging_pipeline, figure,
                 ax, canvas):
    global prev_timestamp
    bmode, metadata = imaging_pipeline(cp.asarray(data), metadata)
    frame_metadata = metadata.custom["frame_metadata_view"][0, :].copy().view(np.int8)
    trigger_counter = frame_metadata[0:8].view(np.uint64).item()
    timestamp = frame_metadata[8:16].view(np.uint64).item() / 65e6
    pulse_counter = frame_metadata[16:20].view(np.uint32).item()
    outa_counter = frame_metadata[20:24].view(np.uint32).item()
    outb_counter = frame_metadata[24:28].view(np.uint32).item()
    canvas.set_data(bmode)
    ax.set_aspect("auto")

    diff = timestamp - prev_timestamp
    prev_timestamp = timestamp
    ax.set_xlabel(f"OX,\n frame: {frame_number}, "
                  f"\n trigger counter: {trigger_counter}, " +
                  "timestamp: %.3f (diff: %.5f), pulse: %d, " % (timestamp, diff, pulse_counter) +
                  f"outa: {outa_counter}, "
                  f"outb: {outb_counter}")
    figure.canvas.flush_events()
    plt.draw()


def acquire_data(cfg_path, batch_size, output_directory, timestamp):
    # Here starts communication with the device.
    session = arrus.session.Session(cfg_path)
    us4r = session.get_device("/Us4R:0")
    # Set initial voltage on the us4r-lite device.
    us4r.set_hv_voltage(VOLTAGE)
    # Upload sequence on the us4r-lite device.
    buffer = us4r.upload(TX_RX_SEQUENCE, mode="sync",
                         host_buffer_size=batch_size,
                         rx_batch_size=batch_size)
    # Start the device.
    us4r.start()
    print("Acquiring data.")
    data, metadata = buffer.tail()
    np.save(os.path.join(output_directory, f"rf_{timestamp}.npy"), data)
    with open(os.path.join(output_directory, f"metadata_{timestamp}.pkl"),
              "wb") as f:
        pickle.dump(metadata, f)
    buffer.release_tail()
    us4r.stop()
    print("Data acquired.")


def main():
    parser = argparse.ArgumentParser(
        description="The script acquires a sequence of RF data and saves "
                    "them to given directory.")

    parser.add_argument("--cfg", dest="cfg",
                        help="Path to session configuration file.",
                        required=True)
    parser.add_argument("--batch_size", dest="batch_size",
                        help="Number of frame to acquire and save.",
                        required=True, type=int)
    parser.add_argument("--output_directory", dest="output_directory",
                        help="Directory where to save the data "
                             "The filename will have a format "
                             "rf_{current time}.npy.",
                        required=True)
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    acquire_data(args.cfg, args.batch_size, args.output_directory, timestamp)

    # Open and reconstruct image.
    session = arrus.session.Session(mock={})
    gpu = session.get_device("/GPU:0")
    x_grid = np.arange(-50, 50, 0.2)*1e-3
    z_grid = np.arange(0, 60, 0.2)*1e-3
    pipeline = Pipeline(
        steps=(
            RemapToLogicalOrder(),
            Transpose(axes=(0, 2, 1)),
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=4, cic_order=2),
            RxBeamforming(),
            EnvelopeDetection(),
            Transpose(),
            ScanConversion(x_grid=x_grid, z_grid=z_grid),
            LogCompression(),
            DynamicRangeAdjustment(min=20, max=80),
            ToGrayscaleImg()),
        placement=gpu)

    fig, ax, canvas = init_display(len(x_grid), len(z_grid))

    print("Loading saved data.")
    data_file_path = os.path.join(args.output_directory,
                                  f"rf_{timestamp}.npy")
    metadata_file_path = os.path.join(args.output_directory,
                                      f"metadata_{timestamp}.pkl")
    batch_data = np.load(data_file_path)
    batch_metadata = pickle.load(open(metadata_file_path, 'rb'))

    print("Displaying the data")
    batch_size = batch_metadata.data_description.custom["frame_channel_mapping"].batch_size
    for i in range(batch_size):
        data = get_batch_data(batch_data, batch_metadata, i)
        metadata = get_batch_metadata(batch_metadata, i)
        display_data(i, data, metadata, pipeline, fig, ax, canvas)


if __name__ == "__main__":
    main()
