"""
Channel-order probe for the data path (Ethernet bench acceptance item, 2026-09-16).

Fires ONE element (TXEL, default 5) with the full receive aperture and reports which RX
columns carry the transmit ringdown. On a correct data path the strongest live column is the
firing element itself; a mirrored row order puts it at 31 - (k mod 32) inside its 32-channel
group, which is what the Ethernet bitstreams announcing status bit 11 did until the RX table
mirror in the us4OEM driver (and the RTL fix marked by status bit 12). The ramp test pattern
cannot see a channel permutation, so this probe is the acceptance check for any change to the
RX data path; run it after the ramp check, on a probe connected to the adapter.

Derived from custom_tx_rx_sequence.py (sample_range shortened to 2048); needs us4r.prototxt
for the connected probe and adapter in the working directory. Usage: TXEL=<k> python3 this.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
K = int(os.environ.get("TXEL", "5"))

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("us4r.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(10)

        n_elements = us4r.get_probe_model().n_elements
        # Full transmit aperture, full receive aperture.
        seq = TxRxSequence(
            ops=[
                TxRx(
                    Tx(aperture=[i == K for i in range(n_elements)],
                       excitation=Pulse(center_frequency=6e6, n_periods=2,
                                        inverse=False),
                       # Custom delays 1.
                       delays=[0]),
                    Rx(aperture=[True]*n_elements,
                       sample_range=(0, 2048),
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
                       sample_range=(0, 2048),
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
        sess.start_scheme()
        for i in range(3):
            data = np.asarray(buffer.get(timeout=180 if i == 0 else 30)[0])
        sess.stop_scheme()
        head = data[:600, :].astype(np.float64)
        energy = (head ** 2).sum(axis=0)
        order = np.argsort(energy)[::-1]
        print(f"TX element {K}: strongest RX columns (energy, first 600 samples): {list(order[:6])}")
        print(f"energy of column {K}: {energy[K]:.3g}, of column {31 - (K % 32) + 32 * (K // 32)}: {energy[31 - (K % 32) + 32 * (K // 32)]:.3g}")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(data[:600, :], aspect="auto", cmap="gray", vmin=-2000, vmax=2000)
        ax.set_title(f"TX element {K} only, RX all; strongest column {order[0]}")
        ax.set_xlabel("RX column (logical element)"); ax.set_ylabel("sample")
        fig.savefig(f"channel_order_probe_txel_{K}.png", dpi=100)
        mirrored = 31 - (K % 32) + 32 * (K // 32)
        verdict = "CORRECT" if energy[K] > energy[mirrored] else "MIRRORED within the 32-channel group"
        print(f"channel order: {verdict} (column {K} vs its mirror {mirrored})")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()
