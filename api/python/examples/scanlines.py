"""
This script acquires and reconstructs RF img for plane wave imaging
(synthetic aperture).

GPU usage is recommended.
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import queue
import numpy as np
import arrus.ops.tgc
import arrus.medium
import cupy as cp

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence,
    DataBufferSpec
)
from arrus.utils.imaging import *
from arrus.utils.gui import (
    Display2D
)

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.TRACE)


class Scanlines(Operation):

    def __init__(self, num_pkg=None, filter_pkg=None):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        self.n_seq, self.n_tx, self.n_samples, self.n_channels = const_metadata.input_shape
        self.output_buffer = cp.zeros((self.n_samples, self.n_tx), dtype=cp.float32)
        return const_metadata.copy(input_shape=(self.n_samples, self.n_tx))

    def process(self, data):
        for i in range(self.n_tx):
            self.output_buffer[:, i] = data[0, i, :, i]
        return self.output_buffer


def main(frequency=15e6):
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("/opt/us4us/us4ndt64.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(30)

        n_elements = us4r.get_probe_model().n_elements
        seq = TxRxSequence(
            ops=[
                TxRx(
                    Tx(aperture=aperture,
                       excitation=Pulse(center_frequency=frequency, n_periods=2,
                                        inverse=False),
                       delays=[0]*np.sum(aperture)),
                    Rx(aperture=[True]*n_elements,
                       sample_range=(0, 6*1024),
                       downsampling_factor=1),
                    pri=1000e-6
                )
                for aperture in np.eye(n_elements, dtype=bool)
            ],
            # Turn off TGC.
            tgc_curve=[],  # [dB]
        )
        # Declare the complete scheme to execute on the devices.
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            rx_buffer_size=4,
            output_buffer=DataBufferSpec(type="FIFO", n_elements=4),
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                    BandpassFilter(order=512), 
                    Scanlines(),
                    Lambda(lambda data: cp.log10(cp.abs(data+1e-9))), 
                ),
                placement="/GPU:0"
            )
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        # us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=34, slope=2e2))
        # Created 2D image display.
        display = Display2D(metadata=metadata, value_range=(-5, 5), aspect="auto")
        # Start the scheme.
        sess.start_scheme()
        # Start the 2D display.
        # The 2D display will consume data put the the input queue.
        # The below function blocks current thread until the window is closed.
        display.start(buffer)

        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()
