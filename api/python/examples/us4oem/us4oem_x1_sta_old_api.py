import time

import matplotlib.pyplot as plt
import numpy as np
import arrus

# Start new session with the device.
sess = arrus.session.InteractiveSession("cfg.yaml")
module = sess.get_device("/Us4OEM:0")
hv256 = sess.get_device("/HV256")
# Configure module's adapter.
interface = arrus.interface.get_interface("esaote")
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
module.set_tgc_samples(np.arange(0.0, 1.0, step=0.2))
module.enable_tgc()

# Configure TX/RX scheme.
NEVENTS = 4
NFRAMES = 121
NSAMPLES = 8*1024
TX_FREQUENCY = 8.125e6
SAMPLING_FREQUENCY = 65e6
NCHANELS = module.get_n_rx_channels()
PRI = 1000e-6 # Pulse Repetition Interval, 1000 [us]

delays = np.array([i * 0.000e-6 for i in range(module.get_n_tx_channels())])

module.clear_scheduled_receive()
module.set_n_triggers(NFRAMES*NEVENTS)
module.set_number_of_firings(NFRAMES*NEVENTS)

for frame in range(NFRAMES):
    for event in range(NEVENTS):
        firing = frame*NEVENTS + event
        tx_aperture = [0] * 128
        tx_aperture[frame] = 1
        tx_aperture[frame+1] = 1
        tx_aperture[frame + 2] = 1
        # tx_aperture[frame + 3] = 1
        # tx_aperture[frame + 4] = 1
        module.set_tx_delays(delays=delays, firing=firing)
        module.set_tx_frequency(frequency=TX_FREQUENCY, firing=firing)
        module.set_tx_half_periods(n_half_periods=3, firing=firing)
        module.set_tx_invert(is_enable=False, firing=firing)
        # module.set_tx_aperture(origin=frame, size=1, firing=firing)
        module.set_tx_aperture_mask(aperture=tx_aperture, firing=firing)
        module.set_active_channel_group([1]*16, firing=firing)
        module.set_rx_time(time=150e-6, firing=firing)
        module.set_rx_delay(delay=5e-6, firing=firing)
        module.set_rx_aperture(origin=event*32, size=32, firing=firing)
        module.schedule_receive(firing*NSAMPLES, NSAMPLES)
        if firing == NFRAMES*NEVENTS-1:
            is_sync_required = True
        else:
            is_sync_required = False
        module.set_trigger(
            time_to_next_trigger=PRI,
            time_to_next_tx=0,
            is_sync_required=is_sync_required,
            idx=firing
        )

module.enable_transmit()

# Run the scheme:
# - prepare figure to display,
rf = np.zeros((NFRAMES, NSAMPLES, module.get_n_rx_channels() * NEVENTS),
              dtype=np.int16)

# - matplotlib related stuff
fig, ax = plt.subplots()
fig.set_size_inches((7, 7))
ax.set_xlabel("Channels")
ax.set_ylabel("Samples")
ax.set_aspect("auto")
fig.canvas.set_window_title("RF data")

# -- create and start canvas
canvas = plt.imshow(
    rf[0, :, :],
    vmin=np.iinfo(np.int16).min,
    vmax=np.iinfo(np.int16).max
)
fig.show()

module.start_trigger()
time.sleep(1)
module.enable_receive()
module.trigger_sync()
# - transfer data from module's internal memory to the host memory
module.stop_trigger()
print("Transferring data")
buffer = module.transfer_rx_buffer_to_host(0, NFRAMES*NEVENTS*NSAMPLES)

for frame_number in range(NFRAMES):
    for firing in range(NEVENTS):
        actual_firing = frame_number * NEVENTS + firing
        rf[frame_number, :, firing * NCHANELS:(firing + 1) * NCHANELS] = \
            buffer[actual_firing * NSAMPLES:(actual_firing + 1) * NSAMPLES, :]

for frame in range(NFRAMES):
    canvas.set_data(rf[frame, :, :])
    ax.set_aspect("auto")
    fig.canvas.flush_events()
    plt.draw()
    time.sleep(0.2)



