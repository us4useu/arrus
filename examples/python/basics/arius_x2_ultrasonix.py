import time

import matplotlib.pyplot as plt
import numpy as np
import arrus

# Start new session with the device.
sess = arrus.session.InteractiveSession("cfg_ultrasonix.yaml")
cards = [sess.get_device("/Arius:0"), sess.get_device("/Arius:1")]

master_card = sess.get_device("/Arius:0")
hv256 = sess.get_device("/HV256")

# Configure module's adapter and start the device.
interface = arrus.interface.get_interface("ultrasonix")
for i, card in enumerate(cards):
    card.store_mappings(
        interface.get_tx_channel_mapping(i),
        interface.get_rx_channel_mapping(i)
    )
    card.start_if_necessary()

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
    card.set_tgc_samples([0x9001]
                      + (0x4000 + np.arange(2500, 0, -50)).tolist()
                      + [0x4000 + 3000])
    card.enable_tgc()

# Configure TX/RX scheme.
NARIUS = 2
NEVENTS = 2
NSAMPLES = 8*1024
TX_FREQUENCY = 8.125e6
PRI = 1000e-6

for card in cards:
    card.set_number_of_firings(NEVENTS)
    card.clear_scheduled_receive()

for i in range(NEVENTS):
    for arius_number in range(NARIUS):
        card = cards[arius_number]
        if arius_number == 0:
            card.set_tx_delays(
                delays=[1e-6]*32
                     + [0.0]*32
                     + [1e-6]*32
                     + [0.0]*32,
                     firing=i)
            card.set_tx_aperture(
                aperture=np.array(
                    [True] * 32
                  + [False]* 32
                  + [True] * 32
                  + [False]* 32
                ),
                firing=i
            )
            if i == 0:
                card.set_rx_aperture(
                    aperture=np.array(
                          [True] * 32
                        + [False] * 32
                        + [False] * 32
                        + [False] * 32
                    ),
                    firing=i
                )
            else:
                card.set_rx_aperture(
                    aperture=np.array(
                          [False] * 32
                        + [False] * 32
                        + [True]  * 32
                        + [False] * 32
                    ),
                    firing=i
                )
        else:
            card.set_tx_delays(
                delays=  [0.0] * 32
                       + [1e-6] * 32
                       + [0.0] * 32
                       + [1e-6] * 32,
                firing=i)
            card.set_tx_aperture(
                aperture=np.array(
                      [False] * 32
                    + [True] * 32
                    + [False] * 32
                    + [True] * 32
                ),
                firing=i
            )
            if i == 0:
                card.set_rx_aperture(
                    aperture=np.array(
                          [False] * 32
                        + [True]  * 32
                        + [False] * 32
                        + [False] * 32
                    ),
                    firing=i
                )
            else:
                card.set_rx_aperture(
                    aperture=np.array(
                          [False] * 32
                        + [False] * 32
                        + [False] * 32
                        + [True]  * 32
                    ),
                    firing=i
                )
        card.set_tx_frequency(frequency=TX_FREQUENCY, firing=i)
        card.set_tx_half_periods(n_half_periods=2, firing=i)
        card.set_tx_invert(is_enable=False, firing=i)
        card.set_rx_time(time=250e-6, firing=i)
        card.set_rx_delay(delay=5e-6, firing=i)
        card.enable_transmit()


for card in cards:
    card.enable_transmit()
    card.clear_scheduled_receive()
    for i in range(NEVENTS):
        card.schedule_receive(i*NSAMPLES, NSAMPLES)
    card.enable_receive()

master_card.set_n_triggers(NEVENTS)

for i in range(NEVENTS):
    master_card.set_trigger(PRI, 0, False, i)
master_card.set_trigger(PRI, 0, True, NEVENTS-1)

# Run the scheme:
# - prepare figure to display
rf = np.zeros((NSAMPLES, 128), dtype=np.int16)

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

while not is_closed:
    start = time.time()
    for card in cards:
        card.enable_receive()
    master_card.trigger_sync()

    # - transfer data from module's internal memory to the host memory
    buffers = []
    for card in cards:
        buffer = card.transfer_rx_buffer_to_host(0, NEVENTS * NSAMPLES)
        buffers.append(buffer)
    # - reorder acquired data
    rf[:, 0:32] = buffers[0][0:NSAMPLES, :]
    rf[:, 32:64] = buffers[1][0:NSAMPLES, :]
    rf[:, 64:96] = buffers[0][NSAMPLES:2*NSAMPLES, :]
    rf[:, 96:128] = buffers[1][NSAMPLES:2*NSAMPLES, :]

    end = time.time()
    print("Acq time: %.3f" % (end - start), end="\r")
    # - display data
    canvas.set_data(rf)
    ax.set_aspect("auto")
    fig.canvas.flush_events()
    plt.draw()

master_card.trigger_stop()


