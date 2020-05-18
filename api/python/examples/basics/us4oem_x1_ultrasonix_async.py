import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import threading
from dataclasses import dataclass

N_FIRINGS = 2


@dataclass
class Display:
    figure: object
    axis: object
    canvas: object
    lock: threading.Lock
    open: bool = True

    def is_open(self):
        with self.lock:
            return open

    def close(self):
        with self.lock:
            self.open = False


def main():
    # Start new session with the device.
    sess = arrus.session.InteractiveSession("cfg.yaml")
    module = sess.get_device("/Us4OEM:0")

    hv256 = sess.get_device("/HV256")
    hv256.enable_hv()
    hv256.set_hv_voltage(20)

    n_samples = 4096
    data_buffer = np.zeros((n_samples, module.get_n_rx_channels()*N_FIRINGS),
                           dtype=np.int16)

    display = init_display(data_buffer)
    configure_module(module)
    run_tx_rx_sequence(
        module,
        n_samples=n_samples, rx_time=100e-6, rx_delay=20e-6,
        tx_frequency=8.125e6,
        pri=200e-6,
        n_half_periods=3,
        callback=lambda _:
            display_rf_data(module, data_buffer, n_samples, display))

    print("Going")
    wait_until_open(display)
    print("Going")
    # module.stop_trigger()
    print("Going")
    time.sleep(5)
    print("Going")


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


def run_tx_rx_sequence(module, n_samples, rx_time, rx_delay,
                       tx_frequency, pri, n_half_periods, callback=None):
    delays = np.array([i*0.000e-6 for i in range(module.get_n_tx_channels())])
    module.clear_scheduled_receive()

    module.set_n_triggers(N_FIRINGS)
    module.set_number_of_firings(N_FIRINGS)

    for firing in range(N_FIRINGS):
        module.set_tx_delays(delays=delays, firing=firing)
        module.set_tx_frequency(frequency=tx_frequency, firing=firing)
        module.set_tx_half_periods(n_half_periods=n_half_periods, firing=firing)
        module.set_tx_invert(is_enable=False)
        module.set_tx_delays(delays, firing=firing)
        module.set_tx_aperture_mask(
            aperture=np.array([1]*32+[0]*32+[1]*32+[0]*32),
            firing=firing
        )
        if firing == 0:
            module.set_rx_aperture_mask(
                aperture=np.array([1]*32+[0]*32+[0]*32+[0]*32),
                firing=firing)
        else:
            module.set_rx_aperture_mask(
                aperture=np.array([0]*32+[0]*32+[1]*32+[0]*32),
                firing=firing)
        module.set_rx_time(time=rx_time, firing=firing)
        module.set_rx_delay(delay=rx_delay, firing=firing)
        # Use the callback for the last firing.
        if callback is not None and firing == (N_FIRINGS-1):
            module.schedule_receive(firing*n_samples, n_samples, callback)
        else:
            module.schedule_receive(firing*n_samples, n_samples)
        module.set_trigger(time_to_next_trigger=pri, time_to_next_tx=0,
                           is_sync_required=False, idx=firing)
    module.enable_transmit()
    # module.set_trigger(pri, 0, True, n_firings-1)
    module.start_trigger()
    module.enable_receive()


def init_display(buffer, window_sizes=(7, 7)):
    fig, ax = plt.subplots()
    fig.set_size_inches(window_sizes)
    ax.set_xlabel("Channels")
    ax.set_ylabel("Samples")
    ax.set_aspect("auto")
    fig.canvas.set_window_title("RF data")
    canvas = plt.imshow(
        buffer,
        vmin=np.iinfo(np.int16).min,
        vmax=np.iinfo(np.int16).max
    )
    display = Display(figure=fig, axis=ax, canvas=canvas,
                      lock=threading.Lock(), open=True)

    def set_closed(_):
        # Polling
        with display.lock:
            display.open = False

    fig.canvas.mpl_connect("close_event", set_closed)
    fig.show()
    return display


def wait_until_open(display: Display):
    plt.show()


def display_rf_data(module, rf_buffer: np.ndarray,
                    n_samples: int, display: Display):
    # Get data from the module.
    buffer = module.transfer_rx_buffer_to_host(0, N_FIRINGS*n_samples)
    # Reorder the data.
    n_channels = module.get_n_rx_channels()
    for firing in range(N_FIRINGS):
        rf_buffer[:, firing*n_channels:(firing+1)*n_channels] = \
            buffer[firing*n_samples:(firing+1)*n_samples, :]
    # Display the data.
    with display.lock:
        is_open = display.open
        if is_open:
            display.canvas.set_data(rf_buffer)
            display.axis.set_aspect("auto")
            display.figure.canvas.flush_events()
            plt.draw()

    # Start the next acquisitions.
    module.enable_receive()

if __name__ == "__main__":
    main()
