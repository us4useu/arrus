"""
Using single Us4OEM to transmit a Single Plane wave and acquire echo data.

In this example:

- we configure Us4OEM,
- we define PWI-like sequence of firings using full Tx aperture
  (all channels active), no tx delay (angle 0),
- run the sequence and acquire a single RF frame.
"""

import matplotlib.pyplot as plt
import numpy as np
import arrus

from arrus.ops import Tx, Rx, TxRx, Sequence, SetHVVoltage
from arrus import SineWave, RegionBasedAperture

from arrus.system import CustomUs4RCfg
from arrus.devices.us4oem import Us4OEMCfg
from arrus.session import SessionCfg


def main():
    # -- DEVICE CONFIGURATION.

    # Prepare system description.
    # Customize this configuration for your setup.
    system_cfg = CustomUs4RCfg(
        n_us4oems=2,
        is_hv256=True,
    )
    # Prepare Us4OEM initial configuration.
    us4oem_cfg = Us4OEMCfg(
        channel_mapping="esaote",
        active_channel_groups=[1]*16,
        dtgc=0,
        active_termination=200
    )

    # -- PROGRAMMING TX/RX SEQUENCE.
    n_samples = 8*1024
    n_firings = 4

    operations = []
    for i in range(n_firings):
        tx = Tx(delays=np.array([i*0.000e-6 for i in range(128)]),
                excitation=SineWave(frequency=8.125e6, n_periods=1.5,
                                    inverse=False),
                aperture=RegionBasedAperture(origin=0,
                                             size=128),
                pri=200e-6)
        rx = Rx(n_samples=n_samples,
                fs_divider=1,
                aperture=RegionBasedAperture(i*32, 32),
                rx_time=160e-6,
                rx_delay=5e-6)
        txrx = TxRx(tx, rx)
        operations.append(txrx)

    tx_rx_sequence = Sequence(operations)

    # -- RUNNING TX/RX SEQUENCE

    # Configure and create communication session with the device.
    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )
    with arrus.Session(cfg=session_cfg) as sess:
        # Use HV256 high voltage supplier.
        hv256 = sess.get_device("/HV256")
        # Get first available Us4OEM module.
        us4oem = sess.get_device("/Us4OEM:0")

        # Set voltage on HV256.
        sess.run(SetHVVoltage(50), feed_dict=dict(device=hv256))

        # Acquire a single RF frame of shape
        # (N_OPERATIONS*N_SAMPLES, N_RX_CHANNELS).
        frame = sess.run(tx_rx_sequence, feed_dict=dict(device=us4oem))
        frame = frame.reshape((n_firings,
                               n_samples,
                               us4oem.get_n_rx_channels()))
        frame = frame.transpose((0, 2, 1))
        frame = frame.reshape((n_firings*us4oem.get_n_rx_channels(),
                               n_samples))

        # Display the data.
        frame = frame.T
        fig, ax = plt.subplots()
        fig.set_size_inches((7, 7))

        ax.set_xlabel("Channels")
        ax.set_ylabel("Samples")
        fig.canvas.set_window_title("RF data")

        plt.imshow(frame,
                   vmin=np.iinfo(np.int16).min,
                   vmax=np.iinfo(np.int16).max)
        ax.set_aspect("auto")
        plt.show()


if __name__ == "__main__":
    main()
