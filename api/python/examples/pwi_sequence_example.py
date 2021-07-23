"""
This script acquires and reconstructs RF img for plane wave imaging
(synthetic aperture).

GPU usage is recommended.
"""


import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import queue
import time
import scipy.signal

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    PwiSequence
)
from arrus.utils.imaging import (
    Pipeline,
    Transpose,
    BandpassFilter,
    FirFilter,
    Decimation,
    QuadratureDemodulation,
    EnvelopeDetection,
    LogCompression,
    Enqueue,
    RxBeamformingImg,
    ReconstructLri,
    Mean
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
    center_frequency = 6e6
    seq = PwiSequence(
        angles=np.linspace(-10, 10, 11)*np.pi/180,
        pulse=Pulse(center_frequency=center_frequency, n_periods=2, inverse=False),
        rx_sample_range=(0, 4096),
        downsampling_factor=1,
        speed_of_sound=1450,
        pri=200e-6,
        sri=50e-3,
        tgc_start=14,
        tgc_slope=2e2)

    display_input_queue = queue.Queue(1)

    x_grid = np.arange(-15, 15, 0.2) * 1e-3
    z_grid = np.arange(5, 45, 0.2) * 1e-3
    taps = scipy.signal.firwin(64, np.array([0.5, 1.5])*center_frequency,
                               pass_zero=False, fs=65e6)

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=4,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=12),
        work_mode="HOST",
        processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Transpose(axes=(0, 2, 1)),
                FirFilter(taps),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                ReconstructLri(x_grid=x_grid, z_grid=z_grid),
                Mean(axis=0),
                EnvelopeDetection(),
                Transpose(),
                LogCompression(),
                Enqueue(display_input_queue, block=False, ignore_full=True)
            ),
            placement="/GPU:0"
        )
    )

    # Here starts communication with the device.
    with arrus.Session(r"C:\Users\Public\us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(30)

        # Upload sequence on the us4r-lite device.
        buffer, const_metadata = sess.upload(scheme)
        display = Display2D(const_metadata, value_range=(20, 80), cmap="gray")
        sess.start_scheme()
        display.start(display_input_queue)
        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()