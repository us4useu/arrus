import scipy.io as sio
import scipy.signal as scs
import matplotlib.pyplot as plt
import numpy as np


def compute_delays(n_samples, n_channels,
                   fs, c,
                   focus, pitch,
                   depth0=0, delay0=17):

    """
    Computes delay matrix for given parameters.

    :param n_samples: number of samples to consider
    :param n_channels: number of channels to consider
    :param fs: transducer's sampling frequency [Hz]
    :param c: speed of sound [m/s]
    :param focus: if float or single element list focus is y focal coordinate.
                  If two element list, its [x,y] focal coordinates.
    :param pitch: transducer's pitch [m]
    :param depth0: the starting depth [m]
    :param delay0: hardware delay
    :return: A delay matrix (nSamp, nChan) [samples]
    """

    # length of transducer
    probe_length = (n_channels - 1) * pitch

    # check if focus is scalar or vector
    if type(focus) == float or len(focus) == 1:
        x_focus = probe_length/2
        y_focus = focus
    else:
        x_focus, y_focus = focus

    # The distance from the line origin along y axis. [m]
    y_grid = np.arange(0, n_samples) / fs * c / 2 + depth0

    # x coordinates of transducer elements
    element_position = np.arange(0, n_channels) * pitch

    # Make y_grid a column vector: (n_samples, 1).
    y_grid = y_grid.reshape((-1, 1))

    # Make element position a row vector: (1, n_channels).
    element_position = element_position.reshape((1, -1))

    # distances between elements and imaged point
    tx_distance = y_grid
    rx_distance = np.sqrt((x_focus - element_position)**2 + y_grid**2)
    total_distance = tx_distance + rx_distance

    # delay related with focusing
    if x_focus > probe_length/2:
        focus_delay = ((x_focus**2 + y_focus**2)**0.5 - y_focus)/c
    else:
        focus_delay = (((probe_length-x_focus)**2 + y_focus**2)**0.5 - y_focus)/c

    path_delay = (total_distance/c)
    delays = path_delay + focus_delay
    delays = delays * fs + 1
    delays = np.round(delays)
    delays += delay0
    delays = delays.astype(int)

    return delays


def beamforming_line(rf, delays):
    """
    Beamforms one line  using delay and sum algorithm.

    :param rf: input RF data of shape (n_samples, n_channels, nLines)
    :param delays: delay matrix of shape (n_samples, n_channels)
    :return: beamformed single RF line
    """

    n_samples, n_channels = delays.shape
    the_line = np.zeros((n_samples))
    for channel in range(n_channels):
        channel_delays = delays[:, channel]
        the_line += rf[channel_delays, channel]

    return the_line


def beamforming_image(rf, delays):
    """
    Beamforms image usign lineBeamform function

    :param rf: input RF data of shape (n_samples, n_channels, n_lines)
    :param delays: delay matrix of shape (n_samples, n_channels)
    :return: beamformed RF image
    """

    n_samples, n_channels = delays.shape
    n_lines = rf.shape[2]
    image = np.zeros((n_samples, n_lines))
    for i_line in range(n_lines):
        rf_line = rf[:, i_line:(i_line+32), i_line]
        this_line = beamforming_line(rf_line, delays)
        image[:, i_line] = this_line

    return image


def calculate_envelope(rf):
    """
    The function calculate envelope using hilbert transform
    :param rf:
    :return: envelope image
    """
    envelope = np.abs(scs.hilbert(rf, axis=0))

    # n_samples, n_lines = rf.shape
    # envelope = np.zeros((n_samples, n_lines))
    # for i_line in range(n_lines):
    #     this_rf_line = rf[:, i_line]
    #     this_envelope_line = np.abs(scs.hilbert(this_rf_line))
    #     envelope[:, i_line] = this_envelope_line

    return envelope


def load_data_pk(path2file, verbose=1):
    """
    This function is for classical beamforming data acquired from simulation no.1,
    (Piotr Karwat).
    The function loads the data and optionally write some info about the data

    :param path2file:
    :param verbose:
    :return: [rf, c, fs, fn, pitch, txFoc, txAp, nElem]
    """
    matlab_data = sio.loadmat(path2file)
    c = matlab_data.get('c0') * 1e-3
    c = np.float(c)

    tx_focus = matlab_data.get('txFoc') * 1e-3
    tx_focus = np.float(tx_focus)

    fs = matlab_data.get('fs')
    fs = np.float(fs)

    fn = matlab_data.get('fn')
    fn = np.float(fn)

    pitch = matlab_data.get('pitch') * 1e-3
    pitch =np.float(pitch)

    n_elements = matlab_data.get('nElem')
    n_elements = np.int(n_elements)

    tx_aperture = matlab_data.get('txAp')
    tx_aperture = np.int(tx_aperture)

    rf = matlab_data.get('rfBfr')

    # Rf data reformat from Piotr Karwat format to more convenient.
    rf = np.transpose(rf, (1, 0, 2))


    if verbose:
        print('imput data keys: ', matlab_data.keys())
        print('speed of sound: ', c)
        print('pitch: ', pitch)
        print('aperture length: ', n_elements)
        print('focal length: ', tx_focus)
        print('subaperture length: ', tx_aperture)

    return [rf, c, fs, fn, pitch, tx_focus, tx_aperture, n_elements]

def amp2dB(image):
    max_image_value = np.max(image)
    image_dB = np.log10(image / max_image_value) * 20
    return image_dB


def make_bmode_image(rf_image, pitch=0.3048*1e-3, c=1540, depth0=0):

    # calculate envelope
    amplitude_image = calculate_envelope(rf_image)

    # convert do dB
    bmode_image = amp2dB(amplitude_image)

    # calculate ticks and labels
    dy = c/2/fs
    dx = pitch
    n_samples, n_lines = rf_image.shape
    max_depth = (n_samples - 1)*dy + depth0
    image_depth = (n_samples - 1)*dy
    probe_width = (n_lines - 1)*dx
    image_proportion = image_depth/probe_width

    n_xticks = 4
    n_yticks = round(n_xticks * image_proportion)

    xticks = np.linspace(0, n_lines, n_xticks)
    xtickslabels = np.linspace(-probe_width/2, probe_width/2, n_xticks)*1e3
    xtickslabels = np.round(xtickslabels, 1)

    yticks = np.linspace(0, n_samples, n_yticks)
    ytickslabels = np.linspace(depth0, max_depth, n_yticks)*1e3
    ytickslabels = np.round(ytickslabels, 1)

    # calculate data aspect for proper image proportions
    data_aspect = dy/dx

    # show the image
    plt.imshow(bmode_image,
                    interpolation='bicubic',
                    aspect=data_aspect,
                    cmap='gray',
                    vmin=-50, vmax=0
                    )

    plt.xticks(xticks, xtickslabels)
    plt.yticks(yticks, ytickslabels)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel('[dB]', rotation=90)
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.show()


# path do file with the data
# path2datafile = '/home/linuser/us4us/usgData/rfBfr.mat'
path2datafile = '/media/linuser/data01/praca/us4us/us4us_testData/us4us/rfBfr.mat'


# load and reformat data
[rf, c, fs, fn, pitch, tx_focus, tx_aperture, n_elements] = load_data_pk(path2datafile, 0)

# compute delays
delays = compute_delays(3000, tx_aperture, fs, c, [15 * pitch, tx_focus], pitch)

# image beamforming
rf_image = beamforming_image(rf[:, :, 15:-17], delays)

# show bmode image
make_bmode_image(rf_image, pitch, c)