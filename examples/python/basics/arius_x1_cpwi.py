import arius as ar
import numpy as np

# Start new session with the device.
sess = ar.session.InteractiveSession("cfg.yaml")
module = sess.get_device("/Arius:0")
hv = sess.get_device("/HV256")
# Configure module's adapter.
interface = ar.interface.get_interface("esaote")
module.store_mappings(
    interface.get_tx_channel_mapping(0),
    interface.get_rx_channel_mapping(0)
)
# Start the device.
module.start_if_necessary()

hv.enable_hv()
hv.set_hv_voltage(50)

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


def compute_delays():
    # TODO(zklim) compute delays here
    # each row should contains n_tx_delays delays to apply
    delays = np.zeros(shape=(2, module.get_n_tx_channels()))
    delays[1, :] = np.array([i * 0.005e-6 for i in range(module.get_n_tx_channels())])
    return delays


delays_array = compute_delays()
NANGLES = delays_array.shape[0]
NFIRINGS = NEVENTS*NANGLES

module.clear_scheduled_receive()
module.set_n_triggers(NFIRINGS)
module.set_number_of_firings(NFIRINGS)


for i in range(NANGLES):
    for event in range(NEVENTS):
        firing = event + i*NEVENTS
        module.set_tx_delays(delays=delays_array[i, :], firing=firing)
        module.set_tx_frequency(frequency=TX_FREQUENCY, firing=firing)
        module.set_tx_half_periods(n_half_periods=2, firing=firing)
        module.set_tx_invert(is_enable=False, firing=firing)
        module.set_tx_aperture(origin=0, size=128, firing=firing)
        module.set_rx_time(time=200e-6, firing=firing)
        module.set_rx_delay(delay=20e-6, firing=firing)
        module.set_rx_aperture(origin=event*32, size=32, firing=firing)
        module.schedule_receive(firing*NSAMPLES, NSAMPLES)
        module.set_trigger(
            time_to_next_trigger=PRI,
            time_to_next_tx=0,
            is_sync_required=False,
            idx=firing
        )

module.enable_transmit()

module.set_trigger(PRI, 0, True, NFIRINGS-1)

# Run the scheme:
# - prepare figure to display,
rf = np.zeros((NANGLES, NSAMPLES, module.get_n_rx_channels()*NEVENTS), dtype=np.int16)

module.trigger_start()
module.enable_receive()
module.trigger_sync()

# - transfer data from module's internal memory to the host memory
buffer = module.transfer_rx_buffer_to_host(0, NFIRINGS*NSAMPLES)
# - reorder acquired data

for i in range(NANGLES):
    for event in range(NEVENTS):
        firing = event + i*NEVENTS
        rf[i, :, event*NCHANELS:(event+1)*NCHANELS] = \
            buffer[firing*NSAMPLES:(firing+1)*NSAMPLES, :]
        
module.trigger_stop()

np.save("rf.npy", rf)
