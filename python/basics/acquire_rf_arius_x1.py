import arius as ar
import numpy as np
import matplotlib.pyplot as plt
import sys

# Start new session with the device.
sess = ar.session.InteractiveSession()
module = sess.get_device("/Arius:0")
module.start_if_necessary()

# Configure module's adapter.
interface = ar.interface.get_interface("esaote")
module.set_tx_mapping(interface.get_tx_channel_mapping(0))
module.set_rx_mapping(interface.get_rx_channel_mapping(0))

# Configure parameters, that will not change later in the example.
module.set_pga_gain(30) # [dB]
module.set_lpf_cutoff(10e6) # [Hz]
module.set_active_termination(200)
module.set_lna_gain(24) #[dB]
module.set_dtgc(0) # [dB]

# Configure TX/RX scheme.
NEVENTS = 4
NSAMPLES = 8192
delays = np.array([i*0.001e-6 for i in range(module.get_n_tx_channels())])
for i in range(NEVENTS):
    module.set_tx_delays(delays=delays, firing=i)
    module.set_tx_frequency(frequency=5e6, firing=i)
    module.set_tx_periods(n_periods=1, firing=i)
    module.set_tx_aperture(origin=0, size=128, firing=i)

    module.set_rx_time(time=200e-6, firing=i)
    module.set_rx_aperture(origin=i*32, size=32, firing=i)
    module.schedule_receive(i*NSAMPLES, NSAMPLES)

module.set_number_of_firings(NEVENTS-1)
module.enable_transmit()

# Run the scheme:
# - prepare figure to display,
buffer = np.zeros((NEVENTS, NSAMPLES, module.get_n_rx_channels()), dtype=np.int16)
rf = np.zeros((NSAMPLES, module.get_n_rx_channels*NEVENTS), dtype=np.int16)

# - matplotlib related stuff
fig, ax = plt.subplots()
fig.set_size_inches((7, 7))
ax.set_xlabel("Channels")
ax.set_ylabel("Samples")
fig.canvas.set_window_title("RF data")

# -- setting window close event handler
is_closed = False

def set_closed(_):
    global is_closed
    # Intentionally not using threading.Lock (or similar) objects here.
    is_closed = True
fig.canvas.mpl_connect("close_event", set_closed)

# -- creating and starting canvas
canvas = plt.imshow(
    rf,
    vmin=np.iinfo(np.int16).min,
    vmax=np.iinfo(np.int16).max
)
fig.show()

while is_closed:
    module.enable_receive()

    # - start the acquisition
    for i in range(NEVENTS):
        module.sw_trigger()
        module.sw_next_tx()

    # - transfer acquired data
    module.transfer_rx_buffer_to_host(buffer, 0)

    # - reorder acquired data
    for i in range(NEVENTS):
        rf[:, i*32:(i+1)*32] =  buffer[i, :, :]

    # - display data
    canvas.set_data(rf)
    fig.canvas.flush_events()
    plt.draw()


