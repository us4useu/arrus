"""
This script shows how to upload multiple TX/RX sequences, each with
a different TGC curve (set for the whole sequence).

The system cyclically performs Sequence:0, followed by Sequence:1. Each
sequence consists of a single plane wave TX/RX (the same TX and RX),
the sequences differ only in the TGC curve:

- Sequence:0: linear TGC (arrus.ops.tgc.LinearTgc),
- Sequence:1: custom TGC curve, given as (time, value) points
  (arrus.ops.us4r.Tgc).

The RF data acquired by both sequences is displayed side by side
(left: Sequence:0, right: Sequence:1).

NOTE: the TGC curves set for different sequences should have different names;
the name can be used to update the TGC curve after uploading the sequences
(see the us4r.set_tgc call below).

GPU usage is required.
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import cupy as cp
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
    Graph,
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

        def create_sequence(name, tgc):
            # Full transmit aperture, full receive aperture.
            return TxRxSequence(
                ops=[
                    TxRx(
                        Tx(aperture=[True]*n_elements,
                           excitation=Pulse(center_frequency=6e6, n_periods=2,
                                            inverse=False),
                           delays=[0]*n_elements),
                        Rx(aperture=[True]*n_elements,
                           sample_range=(0, n_samples),
                           downsampling_factor=1),
                        pri=200e-6
                    ),
                ],
                # TGC curve for all TX/RXs of this sequence.
                tgc_curve=tgc,
                # Time between consecutive acquisitions of this sequence.
                sri=50e-3,
                name=name
            )

        sequences = [
            create_sequence(
                name="Sequence:0",
                # Linear TGC: start gain [dB], slope [dB/m].
                tgc=LinearTgc(start=14, slope=2e2, name="tgc:0")
            ),
            create_sequence(
                name="Sequence:1",
                # Custom TGC: values [dB] for the given time points [s]
                # (relative to the first acquired sample).
                tgc=Tgc(time=[0, 20e-6, 40e-6, 60e-6, 80e-6],
                        values=[14, 54, 14, 54, 14],
                        name="tgc:1")
            ),
        ]

        # Processing: RF data of each sequence -> RF data of both sequences
        # side by side.
        def rf_pipeline(name):
            return Pipeline(
                steps=(
                    RemapToLogicalOrder(),
                    # -> (n_samples, n_elements)
                    Squeeze(),
                ),
                placement="/GPU:0",
                name=name
            )
        processing = Graph(
            operations={
                rf_pipeline("Rf:0"),
                rf_pipeline("Rf:1"),
                Pipeline(
                    steps=(
                        Lambda(
                            lambda xs: cp.concatenate(xs, axis=1),
                            lambda ms: ms[0].copy(
                                input_shape=(n_samples, 2*n_elements))
                        ),
                    ),
                    placement="/GPU:0",
                    name="Concatenate"
                )
            },
            dependencies={
                # Pipeline name <- TX/RX sequence name
                "Rf:0": "Sequence:0",
                "Rf:1": "Sequence:1",
                "Concatenate": ("Rf:0/Output:0", "Rf:1/Output:0"),
                "Output:0": "Concatenate/Output:0"
            }
        )
        scheme = Scheme(
            tx_rx_sequence=sequences,
            processing=processing
        )
        # Upload the scheme on the us4r-lite device.
        buffer, metadata = sess.upload(scheme)
        # Each TGC curve can be updated after the upload, by its name.
        # NOTE: when there is more than one TGC curve, the new curve should
        # have the same length (the same number of TGC samples)
        # as the previous one.
        # us4r.set_tgc(Tgc(time=[0, 20e-6, 40e-6, 60e-6],
        #                  values=[14, 24, 24, 54],
        #                  name="tgc:1"))
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
