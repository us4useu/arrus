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
from collections import deque
import sys

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
    Transpose,
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D
)

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.TRACE)


def main():
    # Here starts communication with the device.
    #with arrus.Session("C:/Users/public/ate_oemplus.prototxt") as sess:
    with arrus.Session("C:/Users/public/us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        #us4r.set_hv_voltage(5)
        n_elements = us4r.get_probe_model().n_elements


        ops = []

        for i in range(n_elements):
            tx_aperture = [False] * n_elements
            tx_aperture[i] = True
            rx_aperture = [False] * n_elements
            rx_aperture[i] = True
            op = TxRx(
                Tx(aperture=tx_aperture,
                   excitation=Pulse(
                       center_frequency=6e6,
                       n_periods=2,
                       inverse=False),
                   delays=[0]),
                Rx(aperture=rx_aperture,
                   sample_range=[0, 1024],
                   downsampling_factor=1),
                pri=200e-6
            )
            ops.append(op)

        seq = TxRxSequence(
            ops=ops,
            tgc_curve=[14],
            sri = 200e-6 * 192 * 2
        )
        raw_queue = deque(maxlen=3)
        queue = deque(maxlen=3)

        # Declare the complete scheme to execute on the devices.
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    Lambda(lambda data: (raw_queue.append(data.get()), data)[1]),
                    RemapToLogicalOrder(),
                    Lambda(lambda data: (queue.append(data.get()), data)[1]),
                    Transpose(),
                    Squeeze()
                ),
                placement="/GPU:0"
            )
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)

        display = Display2D(metadata=metadata, value_range=(-1000, 1000), cmap="viridis")
        sess.start_scheme()
        display.start(buffer)

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()
