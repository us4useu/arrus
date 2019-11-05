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
