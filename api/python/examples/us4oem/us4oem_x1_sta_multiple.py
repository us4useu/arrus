"""
Using single Us4OEM to perform multiple STA sequences.

In this example:

- we configure Us4OEM,
- we define STA-like sequence of firings using  single-element Tx aperture,
  stride 1,
- run the sequence asynchronously and communicate with the device using
  callback function.
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
from threading import Event
import arrus
from arrus.ops import (
    Tx, Rx, TxRx,
    Sequence,
    SetHVVoltage,
    Loop
)
from arrus import (
    SineWave,
    RegionBasedAperture,
    CustomUs4RCfg,
    Us4OEMCfg,
    SessionCfg
)

# -- CONSTANTS
# Number of firings per frame.
# We set it to 4, because we want to acquire data using all 128 Rx channels.
N_FIRINGS_PER_FRAME = 4
# Number of frames to acquire. A single frame is an output array
# of shape: (N_SAMPLES*N_FRAMES*N_FIRINGS_PER_FRAME, number of Rx channels (32))
N_FRAMES = 128
N_SAMPLES = 4*1024

# Number of sequences to perform. This value means, that
# 20 frames will be acquired, the scripts exits.
N_SEQUENCES = 20
# How often the data should be saved. This value means that data will be
# dumped every 10th frame.
SAVE_DATA_FREQ = 10


# -- GLOBAL VARIABLES
# Currently processed frame.
current_frame = 0
# A list names of files that have been saved.
saved_frames = []

# 'Acquisition stopped' event. Setting it to True means for Main thread
# that acquisition has stopped and data can be displayed.
acq_stopped_event = Event()


def callback(rf):
    """
    A callback function used in this example.

    This function:
    - saves data to numpy array file with frequency ``SAVE_DATA_FREQ``.
    - stops acquisition after ``N_SEQUENCES``.
    """
    global current_frame
    if current_frame > N_SEQUENCES:
        # Inform main thread to stop.
        acq_stopped_event.set()
        # Return False if you don't want to acquire data anymore.
        return False
    if current_frame % SAVE_DATA_FREQ == 0:
        # Save data to file.
        filename = f"frame-{current_frame}.npy"
        print(f"Saving data to file: {filename}")
        np.save(filename, rf)
        saved_frames.append(filename)
    current_frame += 1
    # Continue acquisition.
    return True


def main():
    # -- CONFIGURING DEVICE.

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
    def get_full_rx_aperture(element_number):
        """
        This function creates a sequence of 4 Tx/Rx's with Tx aperture
        containing a single active element ``element_number``.
        The sequence allow to acquire a single frame using 128 Rx channels.
        """
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

    # Create a sequence to acquire RF frame
    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(channel)
        for channel in range(N_FRAMES)
    ])))
    # Run this sequence in a loop.
    sequence_loop = Loop(tx_rx_sequence)

    # -- RUNNING TX/RX SEQUENCE

    # Configure and create communication session with the device.
    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )
    with arrus.Session(cfg=session_cfg) as sess:
        # Get HV256 high voltage supplier.
        hv256 = sess.get_device("/HV256")
        # Get first available Us4OEM module.
        us4oem = sess.get_device("/Us4OEM:0")

        # Set voltage on HV256.
        sess.run(SetHVVoltage(50), feed_dict={"device": hv256})

        # Run sequence loop. Loop is a asynchronous operation, so
        sess.run(sequence_loop, feed_dict={"device": us4oem,
                                           "callback": callback})
        # Wait for "Acquisition stopped" event.
        acq_stopped_event.wait(timeout=60)

        # Display the acquired data.
        for filename in saved_frames:
            rf = np.load(filename)
            display_acquired_frame(rf, us4oem.get_n_rx_channels(), filename)


def display_acquired_frame(rf, n_rx_channels, frame_name, window_sizes=(7, 7)):
    # Reshape acquired data:
    #  from (N_FRAMES * N_FIRING_PER_FRAME * N_SAMPLES, N_RX_CHANNELS)
    #  to (N_FRAMES, N_SAMPLES, N_FIRING_PER_FRAME * N_RX_CHANNELS)
    rf = rf.reshape((N_FRAMES * N_FIRINGS_PER_FRAME,
                     N_SAMPLES,
                     n_rx_channels))
    rf = rf.transpose((0, 2, 1))
    rf = rf.reshape((N_FRAMES,
                     N_FIRINGS_PER_FRAME * n_rx_channels,
                     N_SAMPLES))
    rf = rf.transpose((0, 2, 1))
    fig, ax = plt.subplots()

    # Display the data using matplotlib.
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
