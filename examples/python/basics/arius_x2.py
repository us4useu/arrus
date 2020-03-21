import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import arius

# Start new session with the device.
sess = arius.session.InteractiveSession("cfg.yaml")
cards = [sess.get_device("/Arius:0"), sess.get_device("/Arius:1")]

master_card = sess.get_device("/Arius:0")
events = {
    "Arius:0": [0, 1, 2, 3],
    "Arius:1": [0, 1, 2, 3]
}

hv256 = sess.get_device("/HV256")

# Configure module's adapter and start the device.
interface = arius.interface.get_interface("esaote")
for i, card in enumerate(cards):
    card.store_mappings(
        interface.get_tx_channel_mapping(i),
        interface.get_rx_channel_mapping(i)
    )
    card.start_if_necessary()

try:
    hv256.enable_hv()
    hv256.set_hv_voltage(50)
except RuntimeError:
    print("First try with hv256 didn't work, trying again.")
    hv256.enable_hv()
    hv256.set_hv_voltage(50)

# Configure parameters, that will not change later in the example.
for card in cards:
    card.set_pga_gain(30)  # [dB]
    card.set_lpf_cutoff(15e6)  # [Hz]
    card.set_active_termination(200)
    card.set_lna_gain(24)  # [dB]
    card.set_dtgc(0)
    # card.disable_tgc()
    card.set_tgc_samples([0x9001] + (0x4000 + np.arange(1500, 0, -14)).tolist() + [0x4000 + 3000])
    card.enable_tgc()

# Configure TX/RX scheme.
NEVENTS = 4
NSAMPLES = 8*1024
TX_FREQUENCY = 5.5e6
SAMPLING_FREQUENCY = 65e6
NCHANNELS = master_card.get_n_rx_channels()
PRI = 1000e-6

b, a = scipy.signal.butter(
    2,
    (0.5*TX_FREQUENCY*2/SAMPLING_FREQUENCY, 1.5*TX_FREQUENCY*2/SAMPLING_FREQUENCY),
    'bandpass'
)
tx_channels_set = set(card.get_n_tx_channels() for card in cards)
assert len(tx_channels_set) == 1, "Each card should have the same number of TX channels"
TX_CHANNELS = next(iter(tx_channels_set))
TX_CHANNELS_TOTAL = len(cards)*TX_CHANNELS

delays = np.array([i * 0.000e-6 for i in range(TX_CHANNELS_TOTAL)])

for card in cards:
    card.clear_scheduled_receive()
    card.set_number_of_firings(NEVENTS)

master_card.set_n_triggers(NEVENTS)

for j, card in enumerate(cards):
    offset = 0
    for event in events[card.get_id()]:
        card.set_tx_delays(delays=delays[j*TX_CHANNELS:(j+1)*TX_CHANNELS], firing=event)
        card.set_tx_frequency(frequency=TX_FREQUENCY, firing=event)
        card.set_tx_half_periods(n_half_periods=3, firing=event)
        card.set_tx_invert(is_enable=False)
        card.set_tx_aperture(origin=0, size=128, firing=event)
        card.set_rx_time(time=200e-6, firing=event)
        card.set_rx_delay(delay=20e-6, firing=event)
        card.set_rx_aperture(origin=offset*32, size=32, firing=event)
        card.schedule_receive(offset*NSAMPLES, NSAMPLES)
        offset += 1

for card in cards:
    card.enable_transmit()

for event in events[master_card.get_id()]:
    master_card.set_trigger(
        time_to_next_trigger=PRI,
        time_to_next_tx=0,
        is_sync_required=False,
        idx=event
    )

master_card.set_trigger(PRI, 0, True, events[master_card.get_id()][-1])

# Run the scheme:
# - prepare figure to display
SYS_RX_NCHANNELS = sum(len(events[card.get_id()]) for card in cards) * NCHANNELS
rf = np.zeros((NSAMPLES, SYS_RX_NCHANNELS), dtype=np.int16)

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

master_card.trigger_start()
# time.sleep(0.1)
time.sleep(1*PRI*1e-6*NEVENTS)

while not is_closed:
    start = time.time()
    for card in cards:
        card.enable_receive()
    master_card.trigger_sync()
    time.sleep(1*PRI*1e-6*NEVENTS)

    # - transfer data from module's internal memory to the host memory
    buffers = []
    for card in cards:
        nchunks = len(events[card.get_id()])
        buffer = card.transfer_rx_buffer_to_host(0, nchunks * NSAMPLES)
        buffers.append(buffer)

    # - reorder acquired data
    offset = 0
    for i, card in enumerate(cards):
        buffer = buffers[i]
        for event in events[card.get_id()]:
            rf[:, offset*NCHANNELS:(offset+1)*NCHANNELS] = buffer[event*NSAMPLES:(event+1)*NSAMPLES, :]
            offset += 1

    #scipy.signal.filtfilt(b, a, rf, axis=0)

    end = time.time()
    print("Acq time: %.3f" % (end - start), end="\r")
    # - display data
    canvas.set_data(rf)
    ax.set_aspect("auto")
    fig.canvas.flush_events()
    plt.draw()

master_card.trigger_stop()


