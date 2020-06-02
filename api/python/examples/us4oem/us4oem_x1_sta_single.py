"""
Using single Us4OEM to acquire a single STA sequence.

In this example:

- we configure Us4OEM,
- we define STA-like sequence of firings using single-element Tx aperture,
  stride 1,
- run the sequence and acquire a single RF frame.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import itertools
from arrus.ops import Tx, Rx, TxRx, Sequence, SetHVVoltage
from arrus import SineWave, SingleElementAperture, RegionBasedAperture
from arrus.system import CustomUs4RCfg
from arrus.devices.us4oem import Us4OEMCfg
from arrus.session import SessionCfg


def main():
    # -- DEVICE CONFIGURATION.

    # Prepare system description.
    # Customize this configuration for your setup.
    system_cfg = CustomUs4RCfg(
        n_us4oems=2,
        is_hv256=True
    )
    # Prepare Us4OEM initial configuration.
    us4oem_cfg = Us4OEMCfg(
        channel_mapping="esaote",
        active_channel_groups=[1]*16,
        dtgc=0,
        active_termination=200,
        log_transfer_time=True
    )

    # -- PROGRAMMING TX/RX SEQUENCE.
    n_firings_per_frame = 4
    n_frames = 128
    n_samples = 4*1024

    def get_full_rx_aperture(element_number):
        """
        This function creates a sequence of 4 Tx/Rx's with Tx aperture
        containing a single active element ``element_number``.
        The sequence allow to acquire a single frame using 128 Rx channels.
        """
        operations = []
        for i in range(n_firings_per_frame):
            tx = Tx(excitation=SineWave(frequency=8.125e6, n_periods=1.5,
                                        inverse=False),
                    aperture=RegionBasedAperture(origin=element_number, size=1),
                    pri=300e-6)
            rx = Rx(n_samples=n_samples,
                    fs_divider=2,
                    aperture=RegionBasedAperture(i*32, 32),
                    rx_time=260e-6,
                    rx_delay=5e-6)
            txrx = TxRx(tx, rx)
            operations.append(txrx)
        return operations

    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(channel)
        for channel in range(n_frames)
    ])))

    # -- RUNNING TX/RX SEQUENCE

    # Configure and create communication session with the device.
    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )
    with arrus.Session(cfg=session_cfg) as sess:
        # Enable high voltage supplier.
        hv256 = sess.get_device("/HV256")
        # Get first available Us4OEM module.
        us4oem = sess.get_device("/Us4OEM:0")

        # Set voltage on HV256.
        sess.run(SetHVVoltage(50), feed_dict=dict(device=hv256))

        # Acquire a single RF frame of shape
        # (N_OPERATIONS*N_SAMPLES, N_RX_CHANNELS).
        frame = sess.run(tx_rx_sequence, feed_dict=dict(device=us4oem))

        # Reshape acquired data:
        #  - from (N_FRAMES * N_FIRING_PER_FRAME * N_SAMPLES, N_RX_CHANNELS)
        #    that is: (N_OPERATIONS*N_SAMPLES, N_RX_CHANNELS)
        #  - to (N_FRAMES, N_SAMPLES, N_FIRING_PER_FRAME * N_RX_CHANNELS)
        frame = frame.reshape((n_frames*n_firings_per_frame,
                               n_samples,
                               us4oem.get_n_rx_channels()))
        frame = frame.transpose((0, 2, 1))
        frame = frame.reshape((n_frames,
                               n_firings_per_frame*us4oem.get_n_rx_channels(),
                               n_samples))
        frame = frame.transpose((0, 2, 1))
        # Display the data using matplotlib.
        display_acquired_frame(frame)


def display_acquired_frame(rf, window_sizes=(7, 7)):
    fig, ax = plt.subplots()
    fig.set_size_inches(window_sizes)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Samples")
    fig.canvas.set_window_title("RF data")

    canvas = plt.imshow(rf[0, :, :],
                        vmin=np.iinfo(np.int16).min,
                        vmax=np.iinfo(np.int16).max)
    fig.show()

    for frame_number in range(rf.shape[0]):
        canvas.set_data(rf[frame_number, :, :])
        ax.set_aspect("auto")
        fig.canvas.flush_events()
        ax.set_xlabel(f"Channels (tx: {frame_number})")
        plt.draw()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
