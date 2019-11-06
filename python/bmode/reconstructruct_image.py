import scipy.io as sio
import scipy.signal as scs
import matplotlib.pyplot as plt
import numpy as np

def reconstruct_rf_img(rf, x_grid, z_grid,
                       pitch, fs, fc, pulse_periods, n_first_samples,
                       c, tx_mode, tx_aperture, tx_focus, tx_angle
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


    n_sample, n_channel, n_transmission  = rf.shape
    z_size	= len(z_grid)
    x_size	= len(x_grid)


def load_simulated_data(file):
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

    return [rf, c, fs, fc, pitch, tx_focus, tx_angle, tx_aperture, n_elements, pulse_periods]





file = '/home/linuser/us4us/usgData/rfLin_field.mat'
simdata = load_simulated_data(file)
print(simdata[7])

