import time

import matplotlib.pyplot as plt
import numpy as np
import arrus

# Start new session with the device.
sess = arrus.session.InteractiveSession("cfg_ultrasonix.yaml")
us4oems = [sess.get_device("/Us4OEM:0"), sess.get_device("/Us4OEM:1")]

master_module = sess.get_device("/Us4OEM:0")
hv256 = sess.get_device("/HV256")

# Configure module's adapter and start the device.
interface = arrus.interface.get_interface("ultrasonix")
for i, us4oem in enumerate(us4oems):
    us4oem.store_mappings(
        interface.get_tx_channel_mapping(i),
        interface.get_rx_channel_mapping(i)
    )
    us4oem.start_if_necessary()

hv256.enable_hv()
hv256.set_hv_voltage(20)

# Configure parameters, that will not change later in the example.
for us4oem in us4oems:
    us4oem.set_pga_gain(30)  # [dB]
    us4oem.set_lpf_cutoff(15e6)  # [Hz]
    us4oem.set_active_termination(200)
    us4oem.set_lna_gain(24)  # [dB]
    us4oem.set_dtgc(0)
    # card.disable_tgc()
    us4oem.set_tgc_samples([0x9001]
                           +(0x4000 + np.arange(2500, 0, -50)).tolist()
                           +[0x4000 + 3000])
    us4oem.enable_tgc()

# Configure TX/RX scheme.
NMODULES = 2
NEVENTS = 2
NSAMPLES = 8*1024
TX_FREQUENCY = 8.125e6
PRI = 1000e-6

for us4oem in us4oems:
    us4oem.set_number_of_firings(NEVENTS)
    us4oem.clear_scheduled_receive()

for i in range(NEVENTS):
    for module_number in range(NMODULES):
        us4oem = us4oems[module_number]
        if module_number == 0:
            us4oem.set_tx_delays(
                delays=[1e-6]*32
                     + [0.0]*32
                     + [1e-6]*32
                     + [0.0]*32,
                     firing=i)
            us4oem.set_tx_aperture_mask(
                aperture=np.array(
                    [True] * 32
                  + [False]* 32
                  + [True] * 32
                  + [False]* 32
                ),
                firing=i
            )
            if i == 0:
                us4oem.set_rx_aperture_mask(
                    aperture=np.array(
                          [True] * 32
                        + [False] * 32
                        + [False] * 32
                        + [False] * 32
                    ),
                    firing=i
                )
            else:
                us4oem.set_rx_aperture_mask(
                    aperture=np.array(
                          [False] * 32
                        + [False] * 32
                        + [True]  * 32
                        + [False] * 32
                    ),
                    firing=i
                )
        else:
            us4oem.set_tx_delays(
                delays=  [0.0] * 32
                       + [1e-6] * 32
                       + [0.0] * 32
                       + [1e-6] * 32,
                firing=i)
            us4oem.set_tx_aperture_mask(
                aperture=np.array(
                      [False] * 32
                    + [True] * 32
                    + [False] * 32
                    + [True] * 32
                ),
                firing=i
            )
            if i == 0:
                us4oem.set_rx_aperture_mask(
                    aperture=np.array(
                          [False] * 32
                        + [True]  * 32
                        + [False] * 32
                        + [False] * 32
                    ),
                    firing=i
                )
            else:
                us4oem.set_rx_aperture_mask(
                    aperture=np.array(
                          [False] * 32
                        + [False] * 32
                        + [False] * 32
                        + [True]  * 32
                    ),
                    firing=i
                )
        us4oem.set_tx_frequency(frequency=TX_FREQUENCY, firing=i)
        us4oem.set_tx_half_periods(n_half_periods=2, firing=i)
        us4oem.set_tx_invert(is_enable=False, firing=i)
        us4oem.set_rx_time(time=250e-6, firing=i)
        us4oem.set_rx_delay(delay=5e-6, firing=i)
        us4oem.enable_transmit()


for us4oem in us4oems:
    us4oem.enable_transmit()
    us4oem.clear_scheduled_receive()
    for i in range(NEVENTS):
        us4oem.schedule_receive(i*NSAMPLES, NSAMPLES)
    us4oem.enable_receive()

master_module.set_n_triggers(NEVENTS)

for i in range(NEVENTS):
    master_module.set_trigger(PRI, 0, False, i)
master_module.set_trigger(PRI, 0, True, NEVENTS-1)

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

master_module.trigger_start()

while not is_closed:
    start = time.time()
    for us4oem in us4oems:
        us4oem.enable_receive()
    master_module.trigger_sync()

    # - transfer data from module's internal memory to the host memory
    buffers = []
    for us4oem in us4oems:
        buffer = us4oem.transfer_rx_buffer_to_host(0, NEVENTS*NSAMPLES)
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

master_module.trigger_stop()


