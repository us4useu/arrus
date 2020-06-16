import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from collections.abc import Iterable

def reconstruct_rf_img(rf, x_grid, z_grid,
                       pitch, fs, fc, c,
                       tx_aperture, tx_focus, tx_angle,
                       n_pulse_periods, tx_mode='lin', n_first_samples=0,
                       use_gpu=0,
                       ):
    """
    Function for image reconstruction using delay-and-sum approach.

    :param rf: 3D array of rf signals before beamforming
    :param x_grid: vector of pixel x coordinates [m]
    :param z_grid: vector of pixel z coordinates [m]
    :param pitch: the distance between contiguous elements [m]
    :param fs: sampling frequency [Hz]
    :param fc: carrier frequency [Hz]
    :param c: assumed speed of sound [m/s]
    :param tx_aperture: transmit aperture length [elements]
    :param tx_focus: transmit focus [m]
    :param tx_angle: transmit angle [radians]
    :param n_pulse_periods: the length of the pulse in periods
    :param tx_mode: imaging mode - lin (classical),
                                   sta (synthetic transmit aperture)
                                   pwi (plane wave imaging)
    :param n_first_samples: samples recorded before transmission
    :param use_gpu: if 0 - the cpu is used (default),
                    if 1 - the gpu is used (only for nvidia card with CUDA)
    :return: rf beamformed image

    """

    tx_angle = np.array(tx_angle)
    if tx_angle.size != 1:
        tx_angle = np.squeeze(tx_angle)

    if use_gpu:
        import cupy as cp
        rf = cp.array(rf)
        x_grid = cp.array(x_grid)
        z_grid = cp.array(z_grid)
        tx_angle = cp.array(tx_angle)
        print('recontruction using gpu')
        xp = cp
    else:
        print('recontruction using cpu')
        xp = np

    if tx_focus is None:
        tx_focus = 0

    # making x and z_grid column vector
    z_grid = z_grid[xp.newaxis].T

    # getting size parameters
    n_samples, n_channels, n_transmissions = rf.shape
    z_size = max(z_grid.shape)
    x_size = max(x_grid.shape)

    # check if data is iq (i.e. complex) or 'ordinary' rf (i.e. real)
    is_iqdata = isinstance(rf[0, 0, 0], xp.complex)
    if is_iqdata:
        print('iq (complex) data on input')
    else:
        print('rf (real) data on input')

    # probe/transducer width
    probe_width = (n_channels-1)*pitch

    # x coordinate of transducer elements
    element_xcoord = xp.linspace(-probe_width/2, probe_width/2, n_channels)

    # initial delays [s]
    delay0 = n_first_samples/fs
    burst_factor = 0.5*n_pulse_periods/fc
    is_lin_or_sta = tx_mode == 'lin' or tx_mode == 'sta'
    if is_lin_or_sta and tx_focus > 0:
        focus_delay = (xp.sqrt(((tx_aperture-1)*pitch/2)**2+tx_focus**2)
                      -tx_focus)/c
    else:
        focus_delay = 0

    init_delay = focus_delay+burst_factor+delay0

    # Delay & Sum
    # add zeros as last samples.
    # If a sample is out of range 1: nSamp,
    # then use the sample no.nSamp + 1 which is 0.
    # to be checked if it is faster than irregular memory access.
    tail = xp.zeros((1, n_channels, n_transmissions))
    rf = xp.concatenate((rf, tail))

    # buffers allocation
    rf_tx = xp.zeros((z_size, x_size, n_transmissions))
    if is_iqdata:
        rf_tx = rf_tx.astype(complex)

    weight_tx = xp.zeros((z_size, x_size, n_transmissions))

    # loop over transmissions
    for itx in range(0, n_transmissions):

        # calculate tx delays and apodization

        # classical linear scanning
        # (only a narrow stripe is reconstructed  at a time, no tx apodization)
        if tx_mode == 'lin':

            # difference between image point x coordinate and element x coord
            xdifference = xp.array(x_grid-element_xcoord[itx])

            # logical indexes of valid x coordinates
            lix_valid = (xdifference > (-pitch/2)) & (xdifference <= (pitch/2))
            n_valid = xp.sum(lix_valid)
            n_valid = int(n_valid)

            # ix_valid = list(np.nonzero(lix_valid))
            tx_distance = xp.tile(z_grid, (1, n_valid))
            tx_apodization = xp.ones((z_size, n_valid))

        # synthetic transmit aperture method
        elif tx_mode == 'sta':
            lix_valid = xp.ones(x_size, dtype=bool)
            tx_distance = xp.sqrt((z_grid-tx_focus)**2
                                + (x_grid-element_xcoord[itx])**2
                                )

            tx_distance = tx_distance*xp.sign(z_grid-tx_focus) + tx_focus

            f_number = max(abs(z_grid-tx_focus))
            f_number = max(f_number, xp.array(1e-12))\
                     /abs(x_grid-element_xcoord[itx])*0.5

            tx_apodization = f_number > 2

        elif tx_mode == 'pwi':
            lix_valid = xp.ones((x_size), dtype=bool)

            if tx_angle[itx] >= 0:
                first_element = 0
            else:
                first_element = n_channels-1

            tx_distance = \
                (x_grid-element_xcoord[first_element])*xp.sin(tx_angle[itx]) \
                +z_grid*xp.cos(tx_angle[itx])

            r1 = (x_grid-element_xcoord[0])*xp.cos(tx_angle[itx]) \
                 -z_grid*xp.sin(tx_angle[itx])

            r2 = (x_grid-element_xcoord[-1])*xp.cos(tx_angle[itx]) \
                 -z_grid*xp.sin(tx_angle[itx])

            tx_apodization = (r1 >= 0) & (r2 <= 0)

        else:
            raise ValueError('unknown reconstruction mode!')

        # buffers allocation
        rf_rx = xp.zeros((z_size, x_size, n_channels))
        if is_iqdata:
            rf_rx = rf_rx.astype(complex)

        weight_rx = xp.zeros((z_size, x_size, n_channels))

        # loop over elements
        for irx in range(0, n_channels):

            # calculate rx delays and apodization
            rx_distance = xp.sqrt((x_grid[lix_valid]-element_xcoord[irx])**2
                                  + z_grid**2)
            f_number = abs(z_grid/(x_grid[lix_valid]-element_xcoord[irx])*0.5)
            rx_apodization = f_number > 2

            # calculate total delays [s]
            delays = init_delay + (tx_distance+rx_distance)/c

            # calculate sample number to be used in reconstruction
            samples = delays*fs+1
            out_of_range = (0 > samples) | (samples > n_samples-1)
            samples[out_of_range] = n_samples

            # calculate rf samples (interpolated) and apodization weights
            rf_raw_line = rf[:, irx, itx]
            ceil_samples = xp.ceil(samples).astype(int)
            floor_samples = xp.floor(samples).astype(int)
            valid = xp.where(lix_valid)[0].tolist()
            rf_rx[:, valid, irx] = rf_raw_line[floor_samples]*(1-(samples % 1)) \
                                   + rf_raw_line[ceil_samples]*(samples % 1)
            weight_rx[:, valid, irx] = tx_apodization*rx_apodization

            # modulate if iq signal is used
            if is_iqdata:
                rf_rx[:, lix_valid, irx] = rf_rx[:, lix_valid, irx] \
                                          * xp.exp(1j*2*xp.pi*fc*delays)
                pass

        # calculate rf and weights for single tx
        rf_tx[:, :, itx] = xp.sum(rf_rx*weight_rx, axis=2)
        sumwrx = xp.sum(weight_rx, axis=2)
        weight_tx[:, :, itx] = xp.divide(1, sumwrx,
                                         out=xp.zeros_like(sumwrx),
                                         where=sumwrx!=0)

        # show progress
        percentage = round((itx+1)/n_transmissions*1000)/10
        if itx == 0:
            print('{}%'.format(percentage), end='')
        elif itx == n_transmissions-1:
            print('\r{}%'.format(percentage))
        else:
            print('\r{}%'.format(percentage), end='')

    # calculate final rf image
    rf_image = xp.sum(rf_tx, axis=2)*np.sum(weight_tx, axis=2)

    if use_gpu:
        return cp.asnumpy(rf_image)
    else:
        return rf_image


def make_bmode_image(rf_image, x_grid, y_grid, db_range=-60):
    """
    The function for creating b-mode image.
    
    :param rf_image: 2D rf image
    :param x_grid: vector of x coordinates
    :param y_grid: vector of y coordinates
    :param db_range: dynamic range in [dB].
           If int or float, it is the lower bound of dynamic range,
           and upper bound equal 0 is assumed.
           If list or tuple - min and max values are treated
           as bounds of the dynamic range.
    :return:
    """

    if isinstance(db_range, int) or isinstance(db_range, float):
        db_range = [db_range, 0]
        min_db, max_db = db_range
        if min_db >= max_db:
            raise ValueError(
                "Bad db_range: max_db (now  max_db = {}) "
                "should be larger than min_db (now min_db = {})"
                .format(max_db, min_db)
            )

    if isinstance(db_range, Iterable):
        min_db, max_db = db_range
        if min_db >= max_db:
            raise ValueError(
                "Bad db_range: max_db (now  max_db = {}) "
                "should be larger than min_db (now min_db = {})"
                .format(max_db, min_db)
            )

    # check if 'rf' or 'iq' data on input
    is_iqdata = isinstance(rf_image[1, 1], np.complex)

    dx = x_grid[1]-x_grid[0]
    dy = y_grid[1]-y_grid[0]

    # calculate envelope
    if is_iqdata:
        amplitude_image = np.abs(rf_image)
    else:
        amplitude_image = np.abs(signal.hilbert(rf_image, axis=0))

    # convert do dB
    max_image_value = np.max(amplitude_image)
    bmode_image = np.log10(amplitude_image/max_image_value)*20

    # calculate ticks and labels
    n_samples, n_lines = rf_image.shape
    image_height = (n_samples-1)*dy
    image_height = y_grid[-1]-y_grid[0]
    # max_depth = image_depth + depth0
    # max_depth = z_grid[-1]
    # image_width = (n_lines - 1)*dx
    image_width = x_grid[-1]-x_grid[0]
    image_proportion = image_height/image_width

    n_xticks = 4
    n_yticks = int(round(n_xticks*image_proportion))

    xticks = np.linspace(0, n_lines-1, n_xticks)
    xtickslabels = np.linspace(x_grid[0], x_grid[-1], n_xticks)*1e3
    xtickslabels = np.round(xtickslabels, 1)

    yticks = np.linspace(0, n_samples-1, n_yticks)
    ytickslabels = np.linspace(y_grid[0], y_grid[-1], n_yticks)*1e3
    ytickslabels = np.round(ytickslabels, 1)

    # calculate data aspect for proper image proportions
    data_aspect = dy/dx

    # show the image

    plt.imshow(bmode_image,
               interpolation='bicubic',
               aspect=data_aspect,
               cmap='gray',
               vmin=db_range[0], vmax=db_range[1]
               )

    plt.xticks(xticks, xtickslabels)
    plt.yticks(yticks, ytickslabels)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad=10
    cbar.ax.set_ylabel('[dB]', rotation=90)
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
#    plt.show()


def compute_tx_delays(angles, focus, pitch, c=1490, n_chanels=128):
    """
    Computes Tx delays using given parameters.
    
    
    :param angles: Transmission angles [rad].
                   Can be a number or a list (for multiple angles).
    :param focus: Focal length [m].
    :param pitch: Pitch [m]
    :param c: Speed of sound [m/s]. Default value is 1490.
    :param n_chanels: Number of channels/transducers. Default value is 128.
    :return: Ndarray of delays.
             Its shape is (number of angles, number of channels).
    """

    # transducer indexes
    x_i = np.linspace(0, n_chanels-1, n_chanels)

    # transducer coordinates
    x_c = x_i*pitch

    angles = np.array(angles)
    n_angles = angles.size
    if n_angles != 0:
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


def calculate_envelope(rf):
    """
    The function calculate envelope using hilbert transform
    :param rf:
    :return: envelope image
    """
    envelope = np.abs(signal.hilbert(rf, axis=0))
    return envelope


def rf2iq(rf, fc, fs, decimation_factor):
    """
    Demodulation and decimation from rf signal to iq (quadrature) signal.

    :param rf: array of rf signals
    :param fc: carrier frequency
    :param fs: sampling frequency
    :param decimation_factor: decimation factor
    :return: array of decimated iq signals

    """

    s = rf.shape
    n_dim = len(s)
    n_samples = s[0]

    if n_dim > 1:
        n_channels = s[1]
    else:
        n_channels = 1
        rf = rf[..., np.newaxis]

    if n_dim > 2:
        n_transmissions = s[2]
    else:
        n_transmissions = 1
        rf = rf[..., np.newaxis]
    # creating time array
    ts = 1/fs
    t = np.linspace(0, (n_samples-1)*ts, n_samples)
    t = t[..., np.newaxis, np.newaxis]


    # demodulation
    iq = rf*np.exp(0-1j*2*np.pi*fc*t)


    # low-pass filtration (assuming 150% band)
    f_up_cut = fc*1.5/2

    # ir
    filter_order = 8
    b, a = signal.butter(filter_order,
                         f_up_cut,
                         btype='low',
                         analog=False,
                         output='ba',
                         fs=fs
                         )

    # this scaling of amplitude is to make envelopes from iq and rf similar
    iq = 2*signal.filtfilt(b, a, iq, axis=0)

    # decimation
    if decimation_factor > 1:
        iq = signal.decimate(iq, decimation_factor, axis=0)
    else:
        print('decimation factor <= 1, no decimation')

    iq = np.squeeze(iq)

    return iq

