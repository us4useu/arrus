import time

import matplotlib.pyplot as plt
import numpy as np
import arrus

N_FIRINGS = 2
N_SAMPLES = 8192
RX_TIME = 200e-6

def main():
    # Start new session with the device.
    sess = arrus.session.InteractiveSession("cfg.yaml")
    module = sess.get_device("/Us4OEM:0")
    hv256 = sess.get_device("/HV256")

    hv256.enable_hv()
    hv256.set_hv_voltage(20)

    init_display()

    configure_module(module)


    run_tx_rx_sequence(
        module,
        n_samples=N_SAMPLES, rx_time=RX_TIME, rx_delay=20e-6,
        tx_frequency,
        sampling_frequency,
        pri,
        n_half_periods,
        callback=None)



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


def load_tx_rx_sequence(module, n_samples, rx_time, rx_delay,
                        tx_frequency, sampling_frequency, pri, h_half_periods,
                        callback=None):
    n_channels = module.get_n_rx_channels()
    delays = np.array([i * 0.000e-6 for i in range(module.get_n_tx_channels())])
    module.clear_scheduled_receive()

    module.set_n_triggers(N_FIRINGS)
    module.set_number_of_firings(N_FIRINGS)

    for firing in range(N_FIRINGS):
        module.set_tx_delays(delays=delays, firing=firing)
        module.set_tx_frequency(frequency=tx_frequency, firing=firing)
        module.set_tx_half_periods(n_half_periods=3, firing=firing)
        module.set_tx_invert(is_enable=False)
        module.set_tx_delays(delays, firing=firing)
        module.set_tx_aperture_mask(
            aperture=np.array([1]*32 + [0]*32 + [1]*32 + [0]*32),
            firing=firing
        )
        if firing == 0:
            module.set_rx_aperture_mask(
                aperture=np.array([1]*32 + [0]*32 + [0]*32 + [0]*32),
                firing=firing)
        else:
            module.set_rx_aperture_mask(
                aperture=np.array([0]*32 + [0]*32 + [1]*32 + [0]*32),
                firing=firing)
        module.set_rx_time(time=rx_time, firing=firing)
        module.set_rx_delay(delay=rx_delay, firing=firing)
        # Use the callback for the last firing.
        if callback is not None and firing == (N_FIRINGS - 1):
            module.schedule_receive(firing * n_samples, n_samples, callback)
        else:
            module.schedule_receive(firing * n_samples, n_samples)
        module.set_trigger(time_to_next_trigger=pri, time_to_next_tx=0,
                            is_sync_required=False, idx=firing)
    module.enable_transmit()
    # module.set_trigger(pri, 0, True, n_firings-1)
    module.trigger_start()
    module.enable_receive()

def init_display():
    # - matplotlib related stuff
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.set_xlabel("Channels")
    ax.set_ylabel("Samples")
    ax.set_aspect("auto")
    fig.canvas.set_window_title("RF data")
    fig.canvas.mpl_connect("close_event", set_closed)
    # -- create and start canvas

    rf = np.zeros((N_SAMPLES, module.get_n_rx_channels() * NEVENTS), dtype=np.int16)
    canvas = plt.imshow(
        rf,
        vmin=np.iinfo(np.int16).min,
        vmax=np.iinfo(np.int16).max
    )
    fig.show()

def set_closed(_):
    global is_closed
    # Intentionally not using threading.Lock (or similar) objects here.
    is_closed = True

def reorder_data():

    for event in range(NEVENTS):
        rf[:, event*NCHANELS:(event+1) * NCHANELS] = buffer[event * N_SAMPLES:(event + 1) * N_SAMPLES, :]


while not is_closed:
    module.enable_receive()
    module.trigger_sync()

    # - transfer data from module's internal memory to the host memory
    buffer = module.transfer_rx_buffer_to_host(0, NEVENTS * N_SAMPLES)

    # - reorder acquired data


    end = time.time()
    print("Acq time: %.3f" % (end - start), end="\r")
    # - display data
    canvas.set_data(rf)
    ax.set_aspect("auto")
    fig.canvas.flush_events()
    plt.draw()

module.trigger_stop()


