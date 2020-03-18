import arius as ar
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import sys
import time

# Start new session with the device.
sess = ar.session.InteractiveSession("cfg.yaml")
module = sess.get_device("/Arius:0")
hv256 = sess.get_device("/HV256")
# Configure module's adapter.
interface = ar.interface.get_interface("esaote")
module.store_mappings(
    interface.get_tx_channel_mapping(0),
    interface.get_rx_channel_mapping(0)
)
# Start the device.
module.start_if_necessary()

try:
    hv256.enable_hv()
    hv256.set_hv_voltage(50)
except RuntimeError:
    print("First try with hv256 didn't work, trying again.")
    hv256.enable_hv()
    hv256.set_hv_voltage(50)

# Configure parameters, that will not change later in the example.
module.set_pga_gain(30)  # [dB]
module.set_lpf_cutoff(10e6)  # [Hz]
module.set_active_termination(200)
module.set_lna_gain(24)  # [dB]
module.set_dtgc(0)  # [dB]
module.set_tgc_samples([0x9001] + (0x4000 + np.arange(1500, 0, -14)).tolist() + [0x4000 + 3000])
module.enable_tgc()

# Configure TX/RX scheme.
NEVENTS = 4
NSAMPLES = 6*1024
TX_FREQUENCY = 8.125e6
SAMPLING_FREQUENCY = 65e6
NCHANELS = module.get_n_rx_channels()
PRI = 1000e-6 # Pulse Repetition Interval, 1000 [us]

b, a = scipy.signal.butter(
    2,
    (0.5*TX_FREQUENCY*2/SAMPLING_FREQUENCY, 1.5*TX_FREQUENCY*2/SAMPLING_FREQUENCY),
    'bandpass'
)

delays = np.array([i * 0.000e-6 for i in range(module.get_n_tx_channels())])

module.clear_scheduled_receive()
module.set_n_triggers(NEVENTS)
module.set_number_of_firings(NEVENTS)

for event in range(NEVENTS):
    module.set_tx_delays(delays=delays, firing=event)
    module.set_tx_frequency(frequency=TX_FREQUENCY, firing=event)
    module.set_tx_half_periods(n_half_periods=3, firing=event)
    module.set_tx_invert(is_enable=False)
    module.set_tx_aperture(origin=0, size=128, firing=event)
    module.set_rx_time(time=200e-6, firing=event)
    module.set_rx_delay(delay=20e-6, firing=event)
    module.set_rx_aperture(origin=event*32, size=32, firing=event)
    module.schedule_receive(event*NSAMPLES, NSAMPLES)
    module.set_trigger(
        time_to_next_trigger=PRI,
        time_to_next_tx=0,
        is_sync_required=False,
        idx=event
    )

module.enable_transmit()

module.set_trigger(PRI, 0, True, NEVENTS-1)

# Run the scheme:
# - prepare figure to display,
rf = np.zeros((NSAMPLES, module.get_n_rx_channels() * NEVENTS), dtype=np.int16)

# - matplotlib related stuff
fig, ax = plt.subplots()
fig.set_size_inches((7, 7))
ax.set_xlabel("Channels")
ax.set_ylabel("Samples")
ax.set_aspect("auto")
fig.canvas.set_window_title("RF data")

is_closed = False


def set_closed(_):
    global is_closed
    # Intentionally not using threading.Lock (or similar) objects here.
    is_closed = True


fig.canvas.mpl_connect("close_event", set_closed)

# -- create and start canvas
canvas = plt.imshow(
    rf,
    vmin=np.iinfo(np.int16).min,
    vmax=np.iinfo(np.int16).max
)
fig.show()

module.trigger_start()
time.sleep(1*PRI*1e-6*NEVENTS)

while not is_closed:
    start = time.time()
    module.enable_receive()
    module.trigger_sync()
    time.sleep(1*PRI*1e-6*NEVENTS)

    # - transfer data from module's internal memory to the host memory
    buffer = module.transfer_rx_buffer_to_host(0, NEVENTS*NSAMPLES)

    # - reorder acquired data
    for event in range(NEVENTS):
        rf[:, event*NCHANELS:(event+1) * NCHANELS] = buffer[event*NSAMPLES:(event+1) * NSAMPLES, :]

    scipy.signal.filtfilt(b, a, rf, axis=0)

    end = time.time()
    print("Acq time: %.3f" % (end - start), end="\r")
    # - display data
    canvas.set_data(rf)
    ax.set_aspect("auto")
    fig.canvas.flush_events()
    plt.draw()

module.trigger_stop()


