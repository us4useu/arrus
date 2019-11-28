import scipy.io as sio
import scipy.signal as scs
import matplotlib.pyplot as plt
import numpy as np
import time

def compute_delays(n_samples, aperture, fs, c, focus, pitch, depth0=0,
                       delay0=17):
    """
    Computes delay matrix for given parameters.
    Classical scheme (lin)

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
    aperture_length = (aperture-1)*pitch

    # check if focus is scalar or vector
    if type(focus) == float or len(focus) == 1:
        x_focus = aperture_length/2
        y_focus = focus
    else:
        x_focus, y_focus = focus

    # The distance from the line origin along y axis. [m]
    y_grid = np.arange(0, n_samples)/fs*c/2+depth0

    # x coordinates of transducer elements
    element_position = np.arange(0, aperture)*pitch

    # Make y_grid a column vector: (n_samples, 1).
    y_grid = y_grid.reshape((-1, 1))

    # Make element position a row vector: (1, n_channels).
    element_position = element_position.reshape((1, -1))

    # distances between elements and imaged point
    tx_distance = y_grid
    rx_distance = np.sqrt((x_focus-element_position)**2 + y_grid**2)
    total_distance = tx_distance + rx_distance

    # delay related with focusing
    if x_focus > aperture_length/2:
        focus_delay = ((x_focus**2 + y_focus**2)**0.5 - y_focus)/c
    else:
        focus_delay = (((aperture_length - x_focus)**2 + y_focus**2)**0.5 - y_focus)/c

    path_delay = (total_distance/c)
    delays = path_delay + focus_delay
    delays = delays*fs + 1
    delays += delay0
    delays = np.round(delays)
    delays = delays.astype(int)

    return delays


def beamforming_line(rf, delays):
    """
    Beamforms one line  using delay and sum algorithm.

    :param rf: input RF data of shape (n_samples, n_channels, nLines)
    :param delays: delay matrix of shape (n_samples, n_channels)
    :param n_samples: how much samples will be in line
    :return: beamformed single RF line
    """
    n_samples, n_channels = delays.shape
    the_line = np.zeros(n_samples)
    for channel in range(n_channels):
        channel_delays = delays[:, channel]
        # only sample indexes smaller than n_samples are correct,
        # so bad samples are changes for the last samples
        bad_samples_li = channel_delays >= n_samples
        channel_delays[bad_samples_li] = n_samples - 1
        the_line += rf[channel_delays, channel]

    return the_line


def beamforming_image(rf, tx_aperture, fs, c, focus, pitch):
    """
    Beamforms image from conventional tx/rx scheme

    :param rf: input RF data of shape (n_samples, n_channels, n_lines)
    :param delays: delay matrix of shape (n_samples, n_channels)
    :return: beamformed RF image
    """

    n_samples, n_channels, n_lines = rf.shape
    delays = compute_delays(n_samples, tx_aperture, fs, c, focus, pitch,
                              depth0=0, delay0=17)
    image = np.zeros((n_samples, n_lines))
    half_of_aperture = np.ceil(tx_aperture/2).astype(int)

    # left margin
    for i_line in range(0, half_of_aperture):
        rf_line = rf[:, 0:(i_line + half_of_aperture), i_line]
        delays_line = delays[:, (-half_of_aperture - i_line):]
        this_line = beamforming_line(rf_line, delays_line)
        image[:, i_line] = this_line

    # center of the image
    for i_line in range(half_of_aperture, n_channels - half_of_aperture):
        rf_line = rf[:, (i_line - half_of_aperture):(i_line + half_of_aperture), i_line]
        this_line = beamforming_line(rf_line, delays)
        image[:, i_line] = this_line

    # right margin
    iterator = 0
    for i_line in range(n_channels - half_of_aperture, n_channels):
        delays_line = delays[:, 0:(tx_aperture - iterator)]
        iterator += 1
        rf_line = rf[:, (i_line - half_of_aperture):n_channels, i_line]
        this_line = beamforming_line(rf_line, delays_line)
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


def load_simulated_data(file, verbose=1):
    """
    The function loads data from simulation
    :param file: path2file/filename
    :param verbose: if 1 data keys are printed
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

    pulse_periods = matlab_data.get('nPer')
    pulse_periods = np.int(pulse_periods)

    pitch = matlab_data.get('pitch')
    pitch = np.float(pitch)

    if 'txAng' in matlab_data:
        tx_angle = matlab_data.get('txAng')
        tx_angle = np.radians(tx_angle)
        tx_angle = tx_angle.T
    else:
        tx_angle = 0

    if 'txAp' in matlab_data:
        tx_aperture = matlab_data.get('txAp')
        tx_aperture = np.int(tx_aperture)
    else:
        tx_aperture = 1

    if 'txFoc' in matlab_data:
        tx_focus = matlab_data.get('txFoc')
        tx_focus = np.float(tx_focus)
    else:
        tx_focus = 0

    if 'rfLin' in matlab_data:
        rf = matlab_data.get('rfLin')
        mode = 'lin'

    if 'rfPwi' in matlab_data:
        rf = matlab_data.get('rfPwi')
        mode = 'pwi'

    if 'rfSta' in matlab_data:
        rf = matlab_data.get('rfSta')
        mode = 'sta'

    if verbose:
        print('input data keys: ', matlab_data.keys())
        print(' ')
        print(' ')
        print('mode:', mode)
        print('speed of sound: ', c)
        print('sampling frequency: ', fs)
        print('pulse (carrier) frequency: ', fc)
        print('pitch: ', pitch)
        print('aperture length: ', n_elements)
        print('focal length: ', tx_focus)
        print('subaperture length: ', tx_aperture)
        print('transmission angles: ', tx_angle.T)
        print('number of pulse periods: ', pulse_periods)


    # if verbose:
    #     print('imput data keys: ', matlab_data.keys())
    #     print('speed of sound: ', c)
    #     print('pitch: ', pitch)
    #     print('aperture length: ', n_elements)
    #     print('focal length: ', tx_focus)
    #     print('subaperture length: ', tx_aperture)

    return [rf, c, fs, fc, pitch, tx_focus, tx_angle, tx_aperture, n_elements,
            pulse_periods, mode]


def amp2dB(image):
    max_image_value = np.max(image)
    image_dB = np.log10(image / max_image_value) * 20
    return image_dB



def make_bmode_image(rf_image, dx, dy, depth0=0, draw_colorbar=1):
    """
    The function for creating b-mode image
    :param rf_image: 2D rf image
    :param x_grid: vector of x coordinates
    :param y_grid: vector of y coordinates
    :return:
    """

    # calculate envelope
    amplitude_image = calculate_envelope(rf_image)

    # convert do dB
    max_image_value = np.max(amplitude_image)
    bmode_image = np.log10(amplitude_image/max_image_value)*20

    # calculate ticks and labels
    n_samples, n_lines = rf_image.shape

    aperture_width = (n_lines-1)*dx
    x_grid = np.linspace(-aperture_width/2, aperture_width/2, n_lines)
    z_grid = np.linspace(0, (n_samples-1)*dy, n_samples)
    z_grid = z_grid + depth0
    print(z_grid[-1])

    image_height = (n_samples-1)*dy
    image_width = (n_lines - 1)*dx
    image_proportion = image_height/image_width

    n_xticks = 4
    n_yticks = round(n_xticks*image_proportion)

    xticks = np.linspace(0, n_lines-1, n_xticks)
    xtickslabels = np.linspace(x_grid[0], x_grid[-1], n_xticks)*1e3
    xtickslabels = np.round(xtickslabels, 1)

    yticks = np.linspace(0, n_samples-1, n_yticks)
    ytickslabels = np.linspace(z_grid[0], z_grid[-1], n_yticks)*1e3
    ytickslabels = np.round(ytickslabels, 1)

    # calculate data aspect for proper image proportions
    data_aspect = dy/dx

    # show the image
    plt.imshow(bmode_image,
               interpolation='bicubic',
               aspect=data_aspect,
               cmap='gray',
               vmin=-40, vmax=0
               )

    plt.xticks(xticks, xtickslabels)
    plt.yticks(yticks, ytickslabels)

    if draw_colorbar:
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('[dB]', rotation=90)
        plt.xlabel('[mm]')
        plt.ylabel('[mm]')
        plt.show()


###########################################################################
file = '/media/linuser/data01/praca/us4us/' \
           'us4us_testData/dataSets02/rfLin_field.mat'


# load data
[rf, c, fs, fc, pitch,
tx_focus, tx_angle, tx_aperture,
n_elements, pulse_periods, mode] = load_simulated_data(file, 1)

print('beamforming...')
start_time = time.time()
rf_image = beamforming_image(rf, tx_aperture, fs, c, tx_focus, pitch)
end_time = time.time() - start_time
end_time = round(end_time*10)/10
print("--- %s seconds ---" % end_time)

print(tx_focus, c)
dz = c/fs/2
dx = pitch
make_bmode_image(rf_image, dx, dz, draw_colorbar=1)