import numpy as np
import argparse
import matplotlib.pyplot as plt
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse
import arrus.logging
import arrus.utils.us4r
import time
import pickle
from arrus.utils.imaging import (
    Pipeline,
    BandpassFilter,
    QuadratureDemodulation,
    Decimation,
    RxBeamforming,
    EnvelopeDetection,
    Transpose,
    ScanConversion,
    LogCompression
)

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.TRACE)


def display_data(frame_number, data, metadata, imaging_pipeline):
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.imshow(data)
    ax.set_aspect('auto')
    fig.show()


def display_raw_data(data):
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.imshow(data)
    ax.set_aspect('auto')
    fig.show()


def save_raw_data(frame_number, data, metadata):
    arrus.logging.log(arrus.logging.INFO, f"Saving frame {frame_number}")
    np.save(f"rf_{frame_number}.npy", data)
    with open(f"metadata_{frame_number}.pkl", "wb") as file:
        pickle.dump(metadata, file)


def create_bmode_imaging_pipeline(decimation_factor=4, cic_order=2,
                                  x_grid=None, z_grid=None):
    if x_grid is None:
        x_grid = np.arange(-50, 50, 0.4)*1e-3
    if z_grid is None:
        z_grid = np.arange(0, 60, 0.4)*1e-3

    return Pipeline(
        steps=(
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=decimation_factor,
                       cic_order=cic_order),
            RxBeamforming(),
            EnvelopeDetection(),
            Transpose(),
            ScanConversion(x_grid=x_grid, z_grid=z_grid),
            LogCompression()
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
                        required=True, choices=["nop", "save", "img"])
    parser.add_argument("--n", dest="n",
                        help="How many times should the operation be performed.",
                        required=False, type=int, default=100)
    args = parser.parse_args()

    seq = LinSequence(
        tx_aperture_center_element=np.arange(7, 182),
        tx_aperture_size=64,
        tx_focus=30e-3,
        pulse=Pulse(center_frequency=5e6, n_periods=3.5, inverse=False),
        rx_aperture_center_element=np.arange(7, 182),
        rx_aperture_size=64,
        rx_sample_range=(0, 4096),
        pri=100e-6,
        downsampling_factor=1,
        tgc_start=14,
        tgc_slope=2e2,
        speed_of_sound=1490)
    bmode_imaging = create_bmode_imaging_pipeline()

    action_func = {
        "nop":  None,
        "save": save_raw_data,
        "img":  lambda frame_number, data, metadata: display_data(frame_number, data, metadata, bmode_imaging)
    }[args.action]

    # Here starts communication with the device.
    session = arrus.session.Session(args.cfg)

    us4r = session.get_device("/Us4R:0")
    gpu = session.get_device("/GPU:0")

    # Set the pipeline to be executed on the GPU
    bmode_imaging.set_placement(gpu)
    # Set initial voltage on the us4r-lite device.
    us4r.set_voltage(30)
    # Upload sequence on the us4r-lite device.
    buffer = us4r.upload(seq, mode="sync")

    # Start the device.
    us4r.start()
    times = []
    arrus.logging.log(arrus.logging.INFO, f"Running {args.n} iterations.")
    for i in range(args.n):
        start = time.time()
        data, metadata = buffer.tail()
        if action_func is not None:
            action_func(i, data, metadata)
        buffer.release_tail()
        times.append(time.time()-start)

    arrus.logging.log(arrus.logging.INFO,
         f"Done, average acquisition + processing time: {np.mean(times)} [s]")

    us4r.stop()


if __name__ == "__main__":
    main()
