import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import itertools

from arrus.operations import Tx, Rx, TxRx, Sequence, SetHVVoltage
from arrus.params import SineWave, SingleElementAperture, RegionBasedAperture


def main():
    # Define TX/RX sequence to perform.
    sine_wave = SineWave(frequency=8.125e6, n_periods=2, inverse=False)

    n_firings_per_frame = 4
    n_samples = 8*1024
    arrus.set_log_level(arrus.DEBUG)

    def get_full_rx_aperture(element_number):
        return [
            TxRx(
                tx=Tx(
                    delays=np.zeros(32),
                    excitation=sine_wave,
                    aperture=RegionBasedAperture(32, 32), # SingleElementAperture(element_number),
                    pri=200e-6),
                rx=Rx(
                    sampling_frequency=65e6,
                    n_samples=n_samples,
                    aperture=RegionBasedAperture(i*32, 32),
                    rx_time=150e-6,
                    rx_delay=5e-6))
            for i in range(n_firings_per_frame)
        ]

    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(2)
        for channel in range(n_firings_per_frame)
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
        sess.run(SetHVVoltage(50), feed_dict=dict(device=hv256))

        module = sess.get_device("/Us4OEM:0")
        n_channels = module.get_n_rx_channels()
        configure_module(module)
        time.sleep(2)
        frame = sess.run(tx_rx_sequence, feed_dict=dict(device=module))

        # Copy and reorganize data from the module.
        rf = np.zeros(
            (n_samples, module.get_n_rx_channels() * n_firings_per_frame),
            dtype=np.int16
        )
        for firing in range(n_firings_per_frame):
            rf[:, firing * n_channels:(firing+1) * n_channels] = \
                frame[firing * n_samples:(firing+1) * n_samples, : ]
        display_acquired_frame(rf)
        # display_waveforms(rf)


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
    module.enable_tgc()
    module.set_pga_gain(30)  # [dB]
    module.set_lna_gain(24)  # [dB]
    # Set TGC samples. 'O' (14dB) means minimum gain, '1' means maximum (54 dB)
    # module.set_tgc_samples(np.arange(0.0, 1.0, step=0.2))
    # module.enable_tgc()
    # Set low-pass filter.
    module.set_lpf_cutoff(10e6)  # [Hz]


def display_acquired_frame(buffer, window_sizes=(7, 7)):
    fig, ax = plt.subplots()
    fig.set_size_inches(window_sizes)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Samples")
    fig.canvas.set_window_title("RF data")

    plt.imshow(
        buffer[:, :],
        vmin=np.iinfo(np.int16).min,
        vmax=np.iinfo(np.int16).max
    )
    ax.set_aspect("auto")
    plt.show()


def display_waveforms(buffer):
    fig, axes = plt.subplots(buffer.shape[1], 1)

    for channel in range(buffer.shape[1]):
        axes[channel].plot(buffer[1000:4000, channel])

    plt.show()


if __name__ == "__main__":
    main()
