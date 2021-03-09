"""
This script acquires and reconstructs RF img for classical imaging scheme
(scanline by scanline).

GPU usage is recommended.
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import time
import numpy as np
import queue
from threading import Thread

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    LinSequence
)
from arrus.utils.imaging import (
    Pipeline,
    Lambda,
    Transpose,
    BandpassFilter,
    Decimation,
    QuadratureDemodulation,
    RxBeamforming,
    EnvelopeDetection,
    LogCompression,
    DynamicRangeAdjustment,
    Enqueue,
    ScanConversion
)
from arrus.utils.us4r import (
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D
)

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

x_grid = np.arange(-15, 15, 0.1) * 1e-3
z_grid = np.arange(0, 40, 0.1) * 1e-3

def main():
    seq = LinSequence(
        tx_aperture_center_element=np.arange(0, 192),
        tx_aperture_size=64,
        tx_focus=24e-3,
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_aperture_center_element=np.arange(0, 192),
        rx_aperture_size=64,
        rx_sample_range=(0, 2048),
        pri=200e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1490,
        sri=50e-3)

    data_queue = queue.Queue(1)

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=4,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=12),
        work_mode="HOST",
        processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Transpose(axes=(0, 2, 1)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                RxBeamforming(),
                EnvelopeDetection(),
                Transpose(),
                ScanConversion(x_grid, z_grid),
                LogCompression(),
                Enqueue(data_queue, block=False, ignore_full=True)
            ),
            placement="/GPU:0"
        )
    )

    # Here starts communication with the device.
    with arrus.Session(r"C:\Users\Public\us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(20)

        # Upload sequence on the us4r-lite device.
        buffer, const_metadata = sess.upload(scheme)
        display = Display2D(const_metadata, value_range=(20, 80), cmap="gray")
        sess.start_scheme()
        display.start(data_queue)
        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")

if __name__ == "__main__":
    main()
