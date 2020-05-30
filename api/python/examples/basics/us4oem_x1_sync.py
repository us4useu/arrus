import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import itertools

from arrus.ops import Tx, Rx, TxRx, Sequence, SetHVVoltage
from arrus import SineWave, SingleElementAperture, RegionBasedAperture


def main():
    # Define TX/RX sequence to perform.
    sine_wave = SineWave(frequency=8.125e6, n_periods=1.5, inverse=False)

    n_firings_per_frame = 4
    n_frames = 120
    n_samples = 8*1024

    def get_full_rx_aperture(element_number):
        operations = []
        for i in range(n_firings_per_frame):
            tx = Tx(delays=np.array([0]*5),
                    excitation=sine_wave,
                    aperture=RegionBasedAperture(origin=element_number, size=5),
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
    cfg = dict(
        nModules=2,
        HV256=True,
        masterModule=0
    )
    with arrus.Session(cfg=cfg) as sess:
        # Enable high voltage supplier.
        hv256 = sess.get_device("/HV256")
        module = sess.get_device("/Us4OEM:0")
        configure_module(module)

        sess.run(SetHVVoltage(50), feed_dict=dict(device=hv256))

        n_channels = module.get_n_rx_channels()
        print("Acquiring data")
        frame = sess.run(tx_rx_sequence, feed_dict=dict(device=module))

        # Copy and reorganize data from the module.
        rf = np.zeros(
            (n_frames,
             n_samples,
             module.get_n_rx_channels() * n_firings_per_frame),
            dtype=np.int16
        )
        print("Restructuring data.")
        for frame_number in range(n_frames):
            for firing in range(n_firings_per_frame):
                actual_firing = frame_number*n_firings_per_frame + firing
                rf[frame_number, :, firing*n_channels:(firing+1)*n_channels] = \
                    frame[actual_firing*n_samples:(actual_firing+1)*n_samples, :]
        print("Displaying data")
        display_acquired_frame(rf)


def configure_module(module):
    """
    Configures module:
    - sets channels appropriate mapping for
    - sets RX parameters: TGC, DTGC, LPF, active termination
    """
    interface = arrus.interface.get_interface("esaote")
    module.store_mappings(
        interface.get_tx_channel_mapping(0),
        interface.get_rx_channel_mapping(0)
    )
    # Start the device.
    module.start_if_necessary()
    # Turn off DTGC
    module.set_dtgc(0)  # [dB]
    # Set TGC range [14-54] dB (PGA+LNA)
    module.set_pga_gain(30)  # [dB]
    module.set_lpf_cutoff(10e6)  # [Hz]
    module.set_active_termination(200)
    module.set_lna_gain(24)  # [dB]
    # Set TGC samples. 'O' (14dB) means minimum gain, '1' means maximum (54 dB)
    module.set_tgc_samples(np.arange(0.0, 1.0, step=0.2))
    module.enable_tgc()
    # module.enable_tgc()
    # Set low-pass filter.


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
