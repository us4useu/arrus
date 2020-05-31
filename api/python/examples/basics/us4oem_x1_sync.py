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
    # Prepare system description.
    system_cfg = CustomUs4RCfg(
        n_us4oems=1
    )
    # Prepare Us4OEM initial configuration.
    us4oem_cfg = Us4OEMCfg(
        channel_mapping="esaote",
        active_channel_groups=[1]*16,
        dtgc=0,
        active_termination=200
    )

    # Define TX/RX sequence.
    n_firings_per_frame = 4
    n_frames = 120
    n_samples = 8*1024

    def get_full_rx_aperture(element_number):
        operations = []
        for i in range(n_firings_per_frame):
            tx = Tx(excitation=SineWave(frequency=8.125e6, n_periods=1.5,
                                        inverse=False),
                    aperture=RegionBasedAperture(origin=element_number, size=1),
                    pri=200e-6)
            rx = Rx(sampling_frequency=65e6,
                    n_samples=n_samples,
                    aperture=RegionBasedAperture(i*32, 32),
                    rx_time=150e-6,
                    rx_delay=5e-6)
            txrx = TxRx(tx, rx)
            operations.append(txrx)
        return operations

    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(channel)
        for channel in range(n_frames)
    ])))

    # Execute the sequence in the session.
    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg
        }
    )
    with arrus.Session(cfg=session_cfg) as sess:
        # Enable high voltage supplier.
        hv256 = sess.get_device("/HV256")
        us4oem = sess.get_device("/Us4OEM:0")

        sess.run(SetHVVoltage(50), feed_dict=dict(device=hv256))

        print("Acquiring data")
        frame = sess.run(tx_rx_sequence, feed_dict=dict(device=us4oem))
        # Reshape frame (128*8192, 32) to (128 frames, 8192 samples, 128 chan.)
        print("Restructuring data.")
        frame = frame.reshape((n_frames*n_firings_per_frame,
                               n_samples,
                               us4oem.get_n_rx_channels()))
        frame = frame.transpose((0, 2, 1))
        frame = frame.reshape((n_frames,
                               n_firings_per_frame*us4oem.get_n_rx_channels(),
                               n_samples))
        frame = frame.transpose((0, 2, 1))
        print("Displaying data")
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
