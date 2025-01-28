"""
This script acquires and reconstructs RF img for plane wave imaging
(synthetic aperture).

When you close the window, the last 3 frames collected will be saved to rf.npy file.

GPU usage is required.
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
from collections import deque

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("us4r_L3-9i-D.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(20)

        n_elements = us4r.get_probe_model().n_elements
        q = deque(maxlen=3)
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
                       sample_range=(0, 4096),
                       downsampling_factor=1),
                    pri=200e-6
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
                    pri=200e-6
                ),
            ],
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
                    Lambda(lambda data: (q.append(data.get()), data)[1]),
                    Squeeze(),
                    SelectFrames([0]),
                    Squeeze(),
                ),
                placement="/GPU:0"
            )
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        # us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=34, slope=2e2))
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=54, slope=0))
        # Created 2D image display.
        display = Display2D(metadata=metadata, value_range=(-100, 100))
        # Start the scheme.
        sess.start_scheme()
        # Start the 2D display.
        # The 2D display will consume data put the the input queue.
        # The below function blocks current thread until the window is closed.
        display.start(buffer)
        print("Display closed, stopping the script.")
        np.save("rf.npy", np.stack(q))

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()
