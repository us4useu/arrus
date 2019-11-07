import scipy.io as sio
import scipy.signal as scs
import matplotlib.pyplot as plt
import numpy as np

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


    n_samples, n_channels, n_transmissions  = rf.shape
    z_size = len(z_grid)
    x_size = len(x_grid)

    # probe/transducer width
    probe_width = (n_channels-1)*pitch

    # coordinate of transducer elements
    element_position = np.linspace(-probe_width/2, probe_width, n_channels)

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

    print('focus_delay: ', focus_delay)
    print('init_delay: ', init_delay)

    # Delay & Sum
    # add zeros as last samples.
    # If a sample is out of range 1: nSamp, then use the sample no.nSamp + 1 which is 0.
    # to be checked if it is faster than irregular memory access.
    tail = np.zeros((1, n_channels, n_transmissions))
    rf = np.concatenate((rf, tail))

    # from matlab
    # some buffers allocation
    rf_tx = np.zeros((z_size, x_size, n_transmissions))
    weight_tx = np.zeros((z_size, x_size, n_transmissions))
    rf_rx = np.zeros((z_size, x_size, n_channels))
    weight_rx = np.zeros((z_size, x_size, n_channels))

    # loop over transmissions
    for itx in range(0, n_transmissions):

        # calculate tx delays and apodization

        # classical linear scanning
        # (only a narrow stripe is reconstructed  at a time, no tx apodization)
        if tx_mode == 'lin':
            print((x_grid-element_position[itx]) )
            x_valid = (-pitch/2) <= (x_grid-element_position[itx]) > (-pitch/2)
            print(x_valid)

        elif tx_mode == 'sta':
            print('!')
        elif tx_mode == 'pwi':
            print('!')
        else:
            print('!')





        # loop over elements
        for irx in range(0, n_channels):
            print('')







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
x_grid = np.linspace(-10*1e-3, 10*1e-3, 16)
z_grid = np.linspace(0, 100*1e-3, 32)


# reconstruct data
recrf = reconstruct_rf_img(rf, x_grid, z_grid,
                           pitch, fs, fc, c,
                           tx_aperture, tx_focus, tx_angle,
                           pulse_periods
                           )
