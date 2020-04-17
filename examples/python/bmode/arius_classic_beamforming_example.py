import arius as ar
import numpy as np
import scipy.signal as scs
import time
# import arius.utils.imaging as arim
import imaging as arim
import matplotlib.pyplot as plt


def compute_delays(angles, focus, pitch, c=1490, n_chanels=128):
    # transducer indexes
    x_i = np.linspace(0, n_chanels-1, n_chanels)

    # transducer coordinates
    x_c = x_i*pitch

    # convert angles to ndarray, angles.shape can not be equal ()
    angles = np.array([angles])
    n_angles = angles.size

    # allocating memory for delays
    delays = np.zeros(shape=(n_angles, n_chanels))

    angles = np.array(angles)
    if angles.size != 0:
        # reducing possible singleton dimensions of 'angles'
        angles = np.squeeze(angles)
        if angles.shape == ():
            angles = np.array([angles])

        # allocating memory for delays
        delays = np.zeros(shape=(n_angles, n_chanels))

        # calculating delays for each angle
        for i_angle in range(0, n_angles):
            this_angle = angles[i_angle]
            this_delays = x_c*np.sin(this_angle)/c
            if this_angle < 0:
                this_delays = this_delays-this_delays[-1]
            delays[i_angle, :] = this_delays
    else:
        delays = np.zeros(shape=(1, n_chanels))

    focus = np.array(focus)
    if focus.size == 0:
        return delays

    elif focus.size == 1:
        xf = (n_chanels-1)*pitch/2
        yf = focus

    elif focus.size == 2:
        xf = focus[0] + (n_chanels-1)*pitch/2
        yf = focus[1]

    else:
        print('Bad focus value, set to [] (plane wave)')
        return delays

    # distance between origin of coordinate system and focus
    s0 = np.sqrt(yf**2+xf**2)
    focus_sign = np.sign(yf)

    # cosinus of the angle between array (y=0) and focus position vector
    if s0 == 0:
        cos_alpha = 0
    else:
        cos_alpha = xf/s0

    # distances between elements and focus
    si = np.sqrt(s0**2 + x_c**2 - 2*s0*x_c*cos_alpha)

    # focusing delays
    delays_foc = (s0-si)/c
    delays_foc = delays_foc*focus_sign

    # set min(delays_foc) as delay==0
    d0 = np.min(delays_foc)
    delays_foc = delays_foc - d0

    # full delays
    delays = delays + delays_foc

    return delays



def apdelays2fulldealys(tx_aperture, ntxchanels, i_line, apdelays_array):
    """
    The function returns delays array for probe full aperture in classic
    beamforming mode.
    The function realize sliding of transmit aperture along probe full aperture.

    :param tx_aperture: int, length of transmit aperture
    :param ntxchanels: int, probe full aperture length
    :param i_line: int, number of image line for which delays are generated
    :param apdelays_array: single transmit aperture delays
    :return: fulldelays: ndarray, delays for full aperture
             i_tx_aperture: int, the length of current tx_aperture for i_line
             i_origin: int, the number of the first element of i_tx_aperture
    """
    apdelays_array = np.squeeze(apdelays_array)
    half_aperture = np.floor(tx_aperture/2)
    i1 = np.maximum(0, half_aperture-i_line)
    i2 = np.minimum(tx_aperture, ntxchanels+half_aperture-i_line)
    i3 = np.maximum(0, -half_aperture+i_line)

    if tx_aperture%2:
        i4 = np.minimum(ntxchanels, half_aperture+i_line+1)
    else:
        i4 = np.minimum(ntxchanels, half_aperture+i_line)

    i1 = int(i1)
    i2 = int(i2)
    i3 = int(i3)
    i4 = int(i4)
    apda_ndx = np.arange(i1, i2)
    fude_ndx = np.arange(i3, i4)
    fulldelays = np.zeros(shape=(ntxchanels,))
    fulldelays[fude_ndx] = apdelays_array[apda_ndx]
    i_tx_aperture = len(apda_ndx)
    return fulldelays, i_tx_aperture, i3



##############################################################################
#
#                              DATA ACQUISITION
#
##############################################################################

# parameters
pitch = 0.245e-3
c = 1450
fs = 65e6
fc = 8.125e6
# for now tx_aperture should be <=32
tx_aperture = 32
tx_focus = 20e-3
pulse_periods = 1
mode = 'lin'
n_first_samples = 240
n_chanels = 128


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
hv.set_hv_voltage(20)

# Configure parameters, that will not change later in the example.
module.set_pga_gain(30)  # [dB]
module.set_lpf_cutoff(10e6)  # [Hz]
module.set_active_termination(200)
module.set_lna_gain(24)  # [dB]
module.set_dtgc(0)  # [dB]
module.set_tgc_samples([0x9001] + (0x4000 + np.arange(1500, 0, -14)).tolist() + [0x4000 + 3000])
module.enable_tgc()

# Configure TX/RX scheme.
NEVENTS = 1
NSAMPLES = 6*1024
TX_FREQUENCY = 8.125e6
SAMPLING_FREQUENCY = 65e6
NRXCHANELS = module.get_n_rx_channels()
NTXCHANELS = module.get_n_tx_channels()
PRI = 1000e-6 # Pulse Repetition Interval, 1000 [us]

angles = 0
focus = 30e-3
delays_array = compute_delays(angles, focus, pitch, c, tx_aperture)
NANGLES = delays_array.shape[0]
NLINES = n_chanels
NFIRINGS = NEVENTS*NLINES

module.clear_scheduled_receive()
module.set_n_triggers(NFIRINGS)
module.set_number_of_firings(NFIRINGS)

for i_line in range(NLINES):
    firing = i_line*NEVENTS
    i_delays, i_tx_aperture, i_origin = \
        apdelays2fulldealys(tx_aperture, NTXCHANELS, i_line, delays_array)
    module.set_tx_delays(delays=i_delays, firing=firing)
    module.set_tx_frequency(frequency=TX_FREQUENCY, firing=firing)
    module.set_tx_half_periods(n_half_periods=2, firing=firing)
    module.set_tx_invert(is_enable=False, firing=firing)
    module.set_tx_aperture(origin=i_origin, size=i_tx_aperture, firing=firing)
    module.set_rx_time(time=200e-6, firing=firing)
    module.set_rx_delay(delay=20e-6, firing=firing)
    module.set_rx_aperture(origin=i_origin, size=i_tx_aperture, firing=firing)
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
rf = np.zeros((NLINES, NSAMPLES, n_chanels), dtype=np.int16)
module.trigger_start()
module.enable_receive()
module.trigger_sync()

# - transfer data from module's internal memory to the host memory
buffer = module.transfer_rx_buffer_to_host(0, NFIRINGS*NSAMPLES)
print('buffer shape: ',buffer.shape)

# - reorder acquired data
for i_line in range(NLINES):
    print('i_line: ',i_line)
    i_delays, i_tx_aperture, i_origin = apdelays2fulldealys(tx_aperture,
                                                            NTXCHANELS,
                                                            i_line,
                                                            delays_array
                                                            )
    b1 = np.maximum(0, i_origin - NTXCHANELS + NRXCHANELS)
    b2 = b1+i_tx_aperture
    rf[i_line, :, i_origin:(i_origin+i_tx_aperture)] = \
          buffer[i_line*NSAMPLES:(i_line+1)*NSAMPLES, b1:b2]

module.trigger_stop()

# np.save("rf.npy", rf)


##############################################################################
#
#                             IMAGE RECONSTRUCTION
#
##############################################################################


rf = np.moveaxis(rf, 0, -1)
# rf = np.swapaxes(rf,1,2)
print('rf size: ', rf.shape)

# rf data filtration
f_cut_up = fc*1.5
f_cut_down = fc*0.5
filter_order = 8
b, a = scs.butter(filter_order,
                     [f_cut_down, f_cut_up],
                     btype='bandpass',
                     analog=False,
                     output='ba',
                     fs=fs
                     )
rf_filt = scs.filtfilt(b, a, rf, axis=0)


# define grid for reconstruction (imaged area)
x_grid = np.linspace(-15*1e-3, 15*1e-3, 128)
z_grid = np.linspace( 10*1e-3, 50.*1e-3, 1024)

# RF image reconstruction
start_time = time.time()
rf_image = arim.reconstruct_rf_img(rf_filt,
                                 x_grid,
                                 z_grid,
                                 pitch,
                                 fs,
                                 fc,
                                 c,
                                 tx_aperture,
                                 tx_focus,
                                 angles,
                                 pulse_periods,
                                 mode,
                                 n_first_samples,
                                 use_gpu=0,
                            )
end_time = time.time() - start_time
end_time = round(end_time*10)/10
print("--- %s seconds ---" % end_time)


# show b-mode image
arim.make_bmode_image(rf_image, x_grid, z_grid,-40)

