"""
This script acquires and reconstructs RF img for single-element synthetic
transmit aperture imaging.

GPU usage is required.
"""
import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import queue

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    StaSequence
)
from arrus.utils.imaging import (
    Pipeline,
    Transpose,
    BandpassFilter,
    Decimation,
    QuadratureDemodulation,
    ReconstructLri,
    EnvelopeDetection,
    LogCompression,
    Enqueue,
    Sum
)
from arrus.utils.us4r import (
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D
)

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    seq = StaSequence(
        tx_aperture_center_element=np.arange(0, 128)+32,
        rx_aperture_center_element=63+32,
        rx_aperture_size=128,
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_sample_range=(0, 2048),
        downsampling_factor=2,
        speed_of_sound=1450,
        pri=200e-6,
        sri=200e-3,
        tgc_start=14,
        tgc_slope=2e2)

    display_input_queue = queue.Queue(1)

    x_grid = np.linspace(-15, 15, 256) * 1e-3
    z_grid = np.linspace(0, 40, 256) * 1e-3

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=2,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=4),
        work_mode="HOST",
        processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Transpose(axes=(0, 2, 1)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                ReconstructLri(x_grid=x_grid, z_grid=z_grid),
                Sum(axis=0),
                EnvelopeDetection(),
                Transpose(),
                LogCompression(),
                Enqueue(display_input_queue, block=False, ignore_full=True),
            ),
            placement="/GPU:0"
        )
    )

    # Here starts communication with the device.
    with arrus.Session(r"C:\Users\Public\us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(50)
        # Upload sequence on the us4r-lite device.
        buffer, const_metadata = sess.upload(scheme)
        display = Display2D(const_metadata, value_range=(20, 80), cmap="gray")
        sess.start_scheme()
        display.start(display_input_queue)

    print("Stopping the example.")


if __name__ == "__main__":
    main()