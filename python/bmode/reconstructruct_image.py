import scipy.io as sio
import scipy.signal as scs
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npml

def reconstruct_rf_img(rf, x_grid, z_grid,
                       pitch, fs, fc, c,
                       tx_aperture, tx_focus, tx_angle,
                       n_pulse_periods, tx_mode='lin', n_first_samples=0,
                       ):

    """
    opis

    rf
    x_grid
    z_grid
    pitch
    fs
    fc
    pulse_periods
    n_first_samples
    c
    tx_mode
    tx_aperture
    tx_focus
    tx_angle




    """

    # making x and z_grid 'vertical vector' (should be more user friendly in future!)
    x_grid = x_grid[np.newaxis]
    temp = z_grid[np.newaxis]
    z_grid = temp.T

    n_samples, n_channels, n_transmissions  = rf.shape
    z_size = len(z_grid)
    x_size = len(x_grid)

    # probe/transducer width
    probe_width = (n_channels-1)*pitch

    # x coordinate of transducer elements
    element_xcoord = np.linspace(-probe_width/2, probe_width, n_channels)

    # initial delays [s]
    delay0 = n_first_samples/fs
    burst_factor = 0.5*n_pulse_periods/fc
    is_lin_or_sta = tx_mode == 'lin' or tx_mode == 'sta'
    if is_lin_or_sta and tx_focus > 0:
        focus_delay = (np.sqrt(((tx_aperture-1)*pitch/2)**2 + tx_focus**2)
                       - tx_focus)/c
    else:
        focus_delay = 0

    init_delay = focus_delay + burst_factor - delay0


    # Delay & Sum
    # add zeros as last samples.
    # If a sample is out of range 1: nSamp, then use the sample no.nSamp + 1 which is 0.
    # to be checked if it is faster than irregular memory access.
    tail = np.zeros((1, n_channels, n_transmissions))
    rf = np.concatenate((rf, tail))

    # buffers allocation
    rf_tx = np.zeros((z_size, x_size, n_transmissions))
    weight_tx = np.zeros((z_size, x_size, n_transmissions))
    # rf_rx = np.zeros((z_size, x_size, n_channels))
    # weight_rx = np.zeros((z_size, x_size, n_channels))

    # loop over transmissions
    for itx in range(0, n_transmissions):

        # calculate tx delays and apodization

        # classical linear scanning
        # (only a narrow stripe is reconstructed  at a time, no tx apodization)
        if tx_mode == 'lin':

            # difference between image point x coordinate and element x coord
            xdifference = np.array(x_grid-element_xcoord[itx])
            # logical indexes of valid x coordinates
            lix_valid = (xdifference >= (-pitch/2)) & (xdifference <= (pitch/2))
            n_valid = np.sum(lix_valid)

            temp = x_grid + z_grid
            tx_distance = npml.repmat(z_grid, 1, n_valid)
            tx_apodization = np.ones((z_size, n_valid))
            # print('itx:', itx, 'tx_distance:',tx_distance,
            #       'tx_apodization', tx_apodization)

        # synthetic transmit aperture method
        elif tx_mode == 'sta':
            lix_valid = np.ones((1, x_size), dtype=bool)
            tx_distance = np.sqrt((z_grid - tx_focus)**2
                                + (x_grid - element_xcoord[itx])**2
                          )
            # WARNING: sign()=0=>invalid txDist value (warning from matlab file)
            tx_distance = tx_distance*np.sign(z_grid - tx_focus) + tx_focus
            f_number = abs(x_grid - element_xcoord[itx])\
                       /max(abs(z_grid - tx_focus), 1e-12)


        elif tx_mode == 'pwi':
            print('todo: pwi !!!')
        else:
            print('unknown reconstruction mode!')

        # buffers allocation
        rf_rx = np.zeros((z_size, x_size, n_channels))
        weight_rx = np.zeros((z_size, x_size, n_channels))



        # loop over elements
        for irx in range(0, n_channels):

            # calculate rx delays and apodization
            rx_distance = np.sqrt((x_grid[lix_valid] - element_xcoord[irx])**2
                                  + z_grid**2)
            f_number = abs(x_grid[lix_valid] - element_xcoord[irx]/z_grid)
            rx_apodization = f_number < 0.5

            # calculate total delays [s]
            delay = init_delay + (tx_distance + rx_distance)/c

            # calculate sample number to be used in reconstruction
            samples = delay*fs + 1
            out_of_range = (0 > samples) & (samples > n_samples)
            samples[out_of_range] = n_samples + 1
            print(samples[out_of_range])

            # calculate rf samples (interpolated) and apodization weights
            rf_raw_line = rf[:, irx, itx]
            ceil_samples = np.ceil(samples).astype(int)
            floor_samples = np.floor(samples).astype(int)
            # print(irx)
            # print(n_samples)
            # print(floor_samples)
            rf_rx = rf_raw_line[floor_samples]*(1 - (samples % 1))\
                    + rf_raw_line[ceil_samples]*(samples % 1)





    return 0


def load_simulated_data(file, verbose=1):
    """

    dokonczyc
    :param file:
    :return:
    """


    matlab_data = sio.loadmat(file)
    c = matlab_data.get('sos')
    c = np.float(c)

    fc = matlab_data.get('fn')
    fc = np.float(fc)

    fs = matlab_data.get('fs')
    fs = np.float(fs)

    n_elements = matlab_data.get('nElem')
    n_elements = np.int(n_elements)

    tx_aperture = matlab_data.get('txAp')
    tx_aperture = np.int(tx_aperture)

    pulse_periods = matlab_data.get('nPer')
    pulse_periods = np.int(pulse_periods)

    pitch = matlab_data.get('pitch')
    pitch = np.float(pitch)

    tx_focus = matlab_data.get('txFoc')
    tx_focus = np.float(tx_focus)

    tx_angle = matlab_data.get('txAng')
    tx_angle = np.int(tx_angle)

    rf = matlab_data.get('rfLin')

    if verbose:
        print('imput data keys: ', matlab_data.keys())
        print(' ')
        print(' ')
        print('speed of sound: ', c)
        print('sampling frequency: ', fs)
        print('pulse (carrier) frequency: ', fc)
        print('pitch: ', pitch)
        print('aperture length: ', n_elements)
        print('focal length: ', tx_focus)
        print('subaperture length: ', tx_aperture)
        print('transmission angles: ', tx_angle)
        print('number of pulse periods: ', pulse_periods)



    return [rf, c, fs, fc, pitch, tx_focus, tx_angle, tx_aperture, n_elements, pulse_periods]


# ippt
# file = '/home/linuser/us4us/usgData/rfLin_field.mat'

# hm
file = '/media/linuser/data01/praca/us4us/' \
       'us4us_testData/dataSets02/rfLin_field.mat'

# load data
[rf, c, fs, fc, pitch,
 tx_focus, tx_angle, tx_aperture,
 n_elements, pulse_periods] = load_simulated_data(file, 0)

# define grid for reconstruction (imaged area)
x_grid = np.linspace(-20*1e-3, 20*1e-3, 512)
z_grid = np.linspace(0, 100*1e-3, 32)


# reconstruct data
recrf = reconstruct_rf_img(rf, x_grid, z_grid,
                           pitch, fs, fc, c,
                           tx_aperture, tx_focus, tx_angle,
                           pulse_periods
                           )
