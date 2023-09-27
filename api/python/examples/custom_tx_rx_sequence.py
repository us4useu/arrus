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
import time

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

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    # Here starts communication with the device.
    with arrus.Session("C:/Users/public/ate.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        #us4r.set_hv_voltage(5)

        n_elements = us4r.get_probe_model().n_elements
        # Full transmit aperture, full receive aperture.
        seq = TxRxSequence(
            ops=[
                TxRx(
                    Tx(aperture=[True]*n_elements,
                       excitation=Pulse(center_frequency=6e6, n_periods=2,
                                        inverse=False),
                       # Custom delays 1.
                       delays=[0]*n_elements),
                    Rx(aperture=[True]*n_elements,
                       sample_range=(0, 2048),
                       downsampling_factor=1),
                    pri=200e-5
                ),
                TxRx(
                    Tx(aperture=[True]*n_elements,
                       excitation=Pulse(center_frequency=6e6, n_periods=2,
                                        inverse=False),
                       # Custom delays 2.
                       delays=np.linspace(0, 1e-6, n_elements)),
                    Rx(aperture=[True]*n_elements,
                       sample_range=(0, 4096),
                       downsampling_factor=1),
                    pri=200e-5
                ),
            ],
            # Turn off TGC.
            tgc_curve=[24]*0,  # [dB]
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

        for n in range (8):
            temp = us4r.read_sequencer(n)
            print("register " + n + " = " + temp)
        #while(1):
        #    print("Display closed, stopping the script.")
        time.sleep(1)

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()
