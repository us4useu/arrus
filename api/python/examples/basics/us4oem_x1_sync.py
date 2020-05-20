import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import itertools

from arrus.operations import Tx, Rx, TxRx, Sequence, SetHVVoltage
from arrus.params import SineWave, SingleElementAperture, RegionBasedAperture


def main():
    # Define TX/RX sequence to perform.
    sine_wave = SineWave(frequency=8.125e6, n_periods=1.5, inverse=False)

    def get_full_rx_aperture(element_number):
        return [
            TxRx(
                tx=Tx(
                    delays=np.zeros(128),
                    excitation=sine_wave,
                    aperture=SingleElementAperture(element_number),
                    pri=200e-6),
                rx=Rx(
                    sampling_frequency=65e-6,
                    n_samples=4096,
                    aperture=RegionBasedAperture(i*32, (i+1)*32)))
            for i in range(4)
        ]

    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(channel)
        for channel in range(128)
    ])))

    # Execute the sequence in the session.
    with arrus.Session(cfg=dict()) as sess:
        hv256 = sess.get_device("/HV256")
        sess.run(SetHVVoltage(50), feed_dict=dict(device=hv256))

        module = sess.get_device("/Us4OEM:0")
        configure_module(module)
        frame = sess.run(tx_rx_sequence, feed_dict=dict(device=module))
        display_acquired_frame(frame)


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
    module.set_lna_gain(24)  # [dB]
    # Set TGC samples. 'O' (14dB) means minimum gain, '1' means maximum (54 dB)
    module.set_tgc_samples(np.arange(0.0, 1.0, step=0.2))
    module.enable_tgc()
    # Set low-pass filter.
    module.set_lpf_cutoff(10e6)  # [Hz]


def display_acquired_frame(buffer, window_sizes=(7, 7)):
    fig, ax = plt.subplots()
    fig.set_size_inches(window_sizes)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Samples")
    fig.canvas.set_window_title("RF data")

    canvas = plt.imshow(
        buffer[0, :, :],
        vmin=np.iinfo(np.int16).min,
        vmax=np.iinfo(np.int16).max
    )
    # Display twice the acquired sequence.
    for _ in range(2):
        for i in range(buffer.shape[0]):
            fig.canvas.set_data(buffer[i, :, :])
            ax.set_aspect("auto")
            fig.canvas.flush_events()
            time.sleep(0.5)
            plt.show()


if __name__ == "__main__":
    main()
