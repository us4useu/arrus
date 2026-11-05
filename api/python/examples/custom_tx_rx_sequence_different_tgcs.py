"""
This script shows how to set a different TGC curve for each TX/RX of
a custom TX/RX sequence.

The sequence consists of three plane wave TX/RXs (the same TX and RX), each
with a different TGC curve:

- TX/RX 0: linear TGC (arrus.ops.tgc.LinearTgc),
- TX/RX 1: custom TGC curve, given as (time, value) points
  (arrus.ops.us4r.Tgc),
- TX/RX 2: constant TGC.

The RF data acquired by the three TX/RXs is displayed side by side
(from left: TX/RX 0, 1, 2).

The TGC curves have names, so they can be updated after uploading the
sequence (see the us4r.set_tgc call below).

GPU usage is required.
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import arrus.ops.tgc
import arrus.medium

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence,
    Tgc
)
from arrus.ops.tgc import LinearTgc
from arrus.utils.imaging import (
    Pipeline,
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
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("/home/pjarosik/us4r.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(10)

        n_elements = us4r.get_probe_model().n_elements
        n_samples = 4096

        # TGC curves [dB], one for each TX/RX.
        tgcs = [
            # Linear TGC: start gain [dB], slope [dB/m]
            LinearTgc(start=34, slope=2e2, name="linear"),
            # Custom TGC: values [dB] for the given time points [s]
            # (relative to the first acquired sample).
            Tgc(time=[0, 20e-6, 40e-6, 80e-6], values=[14, 14, 54, 54], name="custom"),
            # Custom TGC.
            Tgc(time=[0, 30e-6, 60e-6], values=[34, 54, 34], name="constant"),

        ]

        # Full transmit aperture, full receive aperture.
        seq = TxRxSequence(
            ops=[
                TxRx(
                    Tx(aperture=[True]*n_elements,
                       excitation=Pulse(center_frequency=6e6, n_periods=2,
                                        inverse=False),
                       delays=[0]*n_elements),
                    Rx(aperture=[True]*n_elements,
                       sample_range=(0, n_samples),
                       downsampling_factor=1,
                       # TGC curve for this TX/RX.
                       tgc=tgc),
                    pri=200e-6
                )
                for tgc in tgcs
            ],
            # NOTE: tgc_curve (TGC for the whole sequence) should not be set
            # when the TGC is set for each Rx.
            # Time between consecutive acquisitions, i.e. 1/frame rate.
            sri=50e-3
        )
        n_ops = len(seq.ops)
        # Declare the complete scheme to execute on the devices.
        scheme = Scheme(
            # Run the provided sequence.
            tx_rx_sequence=seq,
            # Processing pipeline to perform on the GPU device.
            processing=Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                    Squeeze(),
                    # (n_ops, n_samples, n_elements)
                    # -> (n_samples, n_ops*n_elements), i.e. RF data of
                    # the consecutive TX/RXs side by side.
                    Lambda(
                        lambda data: data.transpose(1, 0, 2).reshape(
                            n_samples, n_ops*n_elements),
                        lambda m: m.copy(
                            input_shape=(n_samples, n_ops*n_elements))
                    ),
                ),
                placement="/GPU:0"
            )
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        # Each TGC curve can be updated after the upload, by its name.
        # NOTE: when there is more than one TGC curve, the new curve should
        # have the same length (the same number of TGC samples)
        # as the previous one.
        # us4r.set_tgc(Tgc(time=[0, 60e-6], values=[24, 24], name="constant"))
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
