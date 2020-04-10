import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import argparse
import parameters as par


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
    tx_angle = np.squeeze(tx_angle)

    if use_gpu:
        rf = cp.array(rf)
        x_grid = cp.array(x_grid)
        z_grid = cp.array(z_grid)
        tx_angle = cp.array(tx_angle)
        print('recontruction using gpu')
    else:
        print('recontruction using cpu')
    xp = cp.get_array_module(rf)


    if tx_focus is None:
        tx_focus = 0


    # making x and z_grid 'vertical vector' (should be more user friendly in future!)
    temp = z_grid[xp.newaxis]
    z_grid = temp.T

    # getting some size parameters
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
    # If a sample is out of range 1: nSamp, then use the sample no.nSamp + 1 which is 0.
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
            # TODO: Should be better image close to the transducer (apodization)

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
                                  +z_grid**2)
            f_number = abs(z_grid/(x_grid[lix_valid]-element_xcoord[irx])*0.5)
            rx_apodization = f_number>2


            # calculate total delays [s]
            delays = init_delay + (tx_distance+rx_distance)/c


            # calculate sample number to be used in reconstruction
            samples = delays*fs+1
            out_of_range = (0>samples)|(samples>n_samples-1)
            samples[out_of_range] = n_samples


            # calculate rf samples (interpolated) and apodization weights
            rf_raw_line = rf[:, irx, itx]
            ceil_samples = xp.ceil(samples).astype(int)
            floor_samples = xp.floor(samples).astype(int)
            valid = xp.where(lix_valid)[0].tolist()
            rf_rx[:, valid, irx] = rf_raw_line[floor_samples]*(1-(samples%1)) \
                                  +rf_raw_line[ceil_samples]*(samples%1)
            weight_rx[:, valid, irx] = tx_apodization*rx_apodization


            # modulate if iq signal is used (to trzeba sprawdzic, bo pisane 'na rybke')
            if is_iqdata:
                # TODO: przetestowac
                rf_rx[:, lix_valid, irx] = rf_rx[:, lix_valid, irx] \
                                          *xp.exp(1j*2*xp.pi*fc*delays)
                pass


        # calculate rf and weights for single tx
        rf_tx[:, :, itx] = xp.sum(rf_rx*weight_rx, axis=2)
        weight_tx[:, :, itx] = xp.sum(weight_rx, axis=2)

        # show progress
        percentage = round((itx+1)/n_transmissions*1000)/10
        if itx == 0:
            print(percentage, '%', end='')
        elif itx == n_transmissions-1:
            print('\r', percentage, '%')
        else:
            print('\r', percentage, '%', end='')

    # calculate final rf image
    rf_image = xp.sum(rf_tx, axis=2)/np.sum(weight_tx, axis=2)

    return cp.asnumpy(rf_image)


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

    if 'rfPwi' in matlab_data:
        rf = matlab_data.get('rfPwi')

    if 'rfSta' in matlab_data:
        rf = matlab_data.get('rfSta')

    if verbose:
        print('input data keys: ', matlab_data.keys())
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

    return [rf, c, fs, fc, pitch, tx_focus, tx_angle, tx_aperture, n_elements,
            pulse_periods]


def calculate_envelope(rf):
    """
    The function calculate envelope using hilbert transform
    :param rf:
    :return: envelope image
    """
    envelope = np.abs(signal.hilbert(rf, axis=0))
    return envelope


def make_bmode_image(rf_image, x_grid, y_grid, dB_threshold=-60):
    """
    The function for creating b-mode image
    :param rf_image: 2D rf image
    :param x_grid: vector of x coordinates
    :param y_grid: vector of y coordinates
    :return:
    """

    # check if 'rf' or 'iq' data on input
    is_iqdata = isinstance(rf_image[1, 1], np.complex)

    dx = x_grid[1]-x_grid[0]
    dy = y_grid[1]-y_grid[0]

    # calculate envelope
    if is_iqdata:
        amplitude_image = np.abs(rf_image)
    else:
        amplitude_image = calculate_envelope(rf_image)

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
               vmin=dB_threshold, vmax=0
               )

    plt.xticks(xticks, xtickslabels)
    plt.yticks(yticks, ytickslabels)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad=10
    cbar.ax.set_ylabel('[dB]', rotation=90)
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.show()


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

    # pomnozenie przez 2 powoduje, ze obwiednie sa takie same z iq i z rf
    iq = 2*signal.filtfilt(b, a, iq, axis=0)

    # decimation
    if decimation_factor > 1:
        iq = signal.decimate(iq, decimation_factor, axis=0)
    else:
        print('decimation factor <= 1, no decimation')

    iq = np.squeeze(iq)

    return iq


def main():
    description_string = 'this file realize image reconstruction \
    from ultrasound data which comes from us4us system'
    parser = argparse.ArgumentParser(description=description_string)

    parser.add_argument("--file", type=str, required=True, default=0,
                        help='The path to the file with 3D array \
                        of pre-beamformed radio-frequency data',
                        dest="file")

    parser.add_argument(
        "--x_grid", dest="x_grid",
        type=float,
        nargs=3,
        help="Definition of interpolation grid along OX axis in [m]. \
        A tuple: (start, stop, number of points) ",
        # default=(-10*1e-3, 10*1e-3, 96),
        required=False)

    parser.add_argument(
        "--z_grid", dest="z_grid",
        type=float,
        nargs=3,
        help="Definition of interpolation grid along OZ axis in [m]. \
        A tuple: (start, stop, number of points). ",
        # default=(5*1e-3, 20*1e-3, 256),
        required=False)

    parser.add_argument(
        "--pitch", dest="pitch",
        type=float,
        required=False,
        # default=0.245e-3,
        help='Distance between neighbouring elements of ultrasound probe, [m]')

    parser.add_argument(
        "--fs", dest="fs",
        type=float,
        required=False,
        # default=65e6,
        help='The sampling frequency in [Hz].')

    parser.add_argument(
        "--fc", dest="fc",
        type=float,
        required=False,
        # default=5e6,
        help='The pulse carrier frequency, [Hz].')

    parser.add_argument(
        "--tx_aperture", dest="tx_aperture",
        type=int,
        required=False,
        # default=192,
        help='Transmit aperture, [number of elements].')

    parser.add_argument(
        "--tx_focus", dest="tx_focus",
        type=float,
        required=False,
        # default=0,
        help='Transmit focus in [m].')

    parser.add_argument(
        "--tx_angle", dest="tx_angle",
        type=float,
        nargs=3,
        required=False,
        # default=(0, 0, 1),
        help='Transmit angles for phased and pwi schemes. \
         A tuple: (start, stop, number of angles), in [deg]')

    parser.add_argument(
        "--n_pulse_periods", dest="n_pulse_periods",
        type=int,
        required=False,
        # default=2,
        help='The number of periods in transmit pulse.')

    parser.add_argument(
        "--tx_mode", dest="tx_mode",
        type=str,
        required=False,
        choices=['lin', 'sta', 'pwi'],
        # default='pwi',
        help='The reconstruction mode. \
        Can be \"pwi\" (plane wave imaging - default) \
        \"lin\" (classic), and \"sta\" (synthetic transmit aperture).')

    parser.add_argument(
        "--n_first_samples", dest="n_first_samples",
        type=int,
        required=False,
        default=315,
        help='The delay from hardware [number of samples].')

    parser.add_argument(
        "--speed_of_sound", dest="c",
        type=float,
        required=False,
        # default=1490,
        help='The assumed speed of sound in the medium, [m/s].')

    parser.add_argument(
        "--use_gpu", dest="use_gpu",
        type=int,
        required=False,
        default=0,
        help='if use_gpu==1, the gpu will be used (cupy).')

    args = parser.parse_args()
    data = par.load_matlab_file(args.file)


    if args.x_grid is None:
        x_grid = np.linspace(-3*1e-3, 3*1e-3, 16)
    else:
        x_grid = args.x_grid

    if args.z_grid is None:
        z_grid = np.linspace(9.5*1e-3, 11.*1e-3, 64)
    else:
        z_grid = args.z_grid


    if args.pitch is None:
        pitch = data[1].pitch
    else:
        pitch = args.pitch


    if args.fs is None:
        fs = data[2].rx.sampling_frequency
    else:
        fs = args.fs


    if args.fc is None:
        fc = data[2].tx.frequency
    else:
        fc = args.fc


    if args.c is None:
        c = data[2].speed_of_sound
    else:
        c = args.c

    if args.tx_aperture is None:
        tx_aperture = data[2].tx.aperture_size
    else:
        tx_aperture = args.tx_aperture


    if args.tx_focus is None:
        tx_focus = data[2].tx.focus
    else:
        tx_focus = args.tx_focus


    if args.tx_angle is None:
        tx_angle = data[2].tx.angles
    else:
        tx_angle = args.tx_angle


    if args.n_pulse_periods is None:
        n_pulse_periods = data[2].tx.n_periods
    else:
        n_pulse_periods = args.n_pulse_periods


    if args.tx_mode is None:
        tx_mode = data[2].mode
    else:
        tx_mode = args.tx_mode


    # reconstruct data
    rf_image = reconstruct_rf_img(data[0],
                                  x_grid,
                                  z_grid,
                                  pitch,
                                  fs,
                                  fc,
                                  c,
                                  tx_aperture,
                                  tx_focus,
                                  tx_angle,
                                  n_pulse_periods,
                                  tx_mode,
                                  n_first_samples=0,
                                  use_gpu=args.use_gpu
                                  )


    make_bmode_image(rf_image, x_grid, z_grid)


    ### with filtration
    # f_cut = [fc*0.25, fc*1.75]
    # fs_img = c/(z_grid[1]-z_grid[0])
    # filter_order = 4
    # b, a = signal.butter(filter_order,
    #                      f_cut, btype='band',
    #                      analog=False,
    #                      output='ba',
    #                      fs=fs_img)
    #
    # rf_image_filt = signal.filtfilt(b, a, rf_image, axis=0)

    # # show image
    # make_bmode_image(rf_image_filt, x_grid, z_grid)


################################################################################


if __name__ == "__main__":
    main()

