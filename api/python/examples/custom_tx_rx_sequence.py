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

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence
)
from arrus.utils.imaging import (
    Pipeline,
    SelectFrames,
    Squeeze,
    Lambda,
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D
)

arrus.set_clog_level(arrus.logging.DEBUG)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("/home/pjarosik/us4r.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(5)

        n_elements = us4r.get_probe_model().n_elements
        # Full transmit aperture, full receive aperture.
        seq = TxRxSequence(
            ops=[
                TxRx(
                    Tx(aperture=np.ones(n_elements).astype(bool),
                       excitation=Pulse(center_frequency=5e6, n_periods=4,
                                        inverse=False),
                       # Custom delays 2.
                       delays=[0]*n_elements),
                    Rx(aperture=np.ones(n_elements).astype(bool), 
                       sample_range=[420, 420+512],
                       downsampling_factor=1),
                    pri=200e-6
                )
             for i in range(64)],
            # Turn off TGC.
            tgc_curve=[],  # [dB]
            # Time between consecutive acquisitions, i.e. 1/frame rate.
            sri=50e-3
        )
        # Declare the complete scheme to execute on the devices.
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                    Squeeze(),
                    SelectFrames([0]),
                    Squeeze(),
                ),
                placement="/GPU:0"
            )
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=34, slope=2e2))
        # Created 2D image display.
        display = Display2D(metadata=metadata, value_range=(-100, 100))
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
