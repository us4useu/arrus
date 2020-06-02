import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import itertools
from threading import Event

from arrus.ops import Tx, Rx, TxRx, Sequence, SetHVVoltage, Loop
from arrus import SineWave, SingleElementAperture, RegionBasedAperture

from arrus.system import CustomUs4RCfg
from arrus.devices.us4oem import Us4OEMCfg
from arrus.session import SessionCfg


NUMBER_OF_REPETITIONS = 20
# How often the data should be saved.
SAVE_FREQUENCY = 10
current_frame = 0
saved_frames = []

N_FIRINGS_PER_FRAME = 4
N_FRAMES = 128
N_SAMPLES = 4*1024

acq_stopped_event = Event()


def callback(rf):
    global current_frame
    if current_frame > NUMBER_OF_REPETITIONS:
        # Finish running the example
        acq_stopped_event.set()
        return False
    if current_frame % SAVE_FREQUENCY == 0:
        filename = f"frame-{current_frame}.npy"
        print(f"Saving data to file: {filename}")
        np.save(filename, rf)
        saved_frames.append(filename)
    current_frame += 1
    return True


def main():
    # Prepare system description.
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

    # Define TX/RX sequence.
    def get_full_rx_aperture(element_number):
        operations = []
        for i in range(N_FIRINGS_PER_FRAME):
            tx = Tx(excitation=SineWave(frequency=8.125e6, n_periods=1.5,
                                        inverse=False),
                    aperture=RegionBasedAperture(origin=element_number, size=1),
                    pri=200e-6)
            rx = Rx(n_samples=N_SAMPLES,
                    fs_divider=2,
                    aperture=RegionBasedAperture(i*32, 32),
                    rx_time=160e-6,
                    rx_delay=5e-6)
            txrx = TxRx(tx, rx)
            operations.append(txrx)
        return operations

    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(channel)
        for channel in range(N_FRAMES)
    ])))

    sequence_loop = Loop(tx_rx_sequence)

    # Execute the sequence in the session.
    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )
    with arrus.Session(cfg=session_cfg) as sess:
        # Enable high voltage supplier.
        hv256 = sess.get_device("/HV256")
        us4oem = sess.get_device("/Us4OEM:0")

        sess.run(SetHVVoltage(50), feed_dict={"device": hv256})

        sess.run(sequence_loop, feed_dict={"device": us4oem,
                                           "callback": callback})
        acq_stopped_event.wait(timeout=60)

        for filename in saved_frames:
            rf = np.load(filename)
            display_acquired_frame(rf, us4oem.get_n_rx_channels(), filename)


def display_acquired_frame(rf, n_rx_channels, frame_name, window_sizes=(7, 7)):
    print("Restructuring data frame.")
    rf = rf.reshape((N_FRAMES * N_FIRINGS_PER_FRAME,
                     N_SAMPLES,
                     n_rx_channels))
    rf = rf.transpose((0, 2, 1))
    rf = rf.reshape((N_FRAMES,
                     N_FIRINGS_PER_FRAME * n_rx_channels,
                     N_SAMPLES))
    rf = rf.transpose((0, 2, 1))
    fig, ax = plt.subplots()
    fig.set_size_inches(window_sizes)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Samples")
    fig.canvas.set_window_title(f"RF data {frame_name}")

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
        time.sleep(0.05)


if __name__ == "__main__":
    main()
