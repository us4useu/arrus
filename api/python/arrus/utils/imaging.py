import numpy as np
import scipy.signal as scs
import matplotlib.pyplot as plt

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
        weight_tx[:, :, itx] = 1/(1+xp.sum(weight_rx, axis=2))

        # show progress
        percentage = round((itx+1)/n_transmissions*1000)/10
        if itx == 0:
            print(percentage, '%', end='')
        elif itx == n_transmissions-1:
            print('\r', percentage, '%')
        else:
            print('\r', percentage, '%', end='')

    # calculate final rf image
    rf_image = xp.sum(rf_tx, axis=2)*np.sum(weight_tx, axis=2)

    if use_gpu:
        return cp.asnumpy(rf_image)
    else:
        return rf_image



def make_bmode_image(rf_image, x_grid, y_grid, db_range=-60):
    """
    The function for creating b-mode image
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
    if isinstance(db_range, int):
        db_range = [db_range, 0]

    elif isinstance(db_range, float):
        db_range = [db_range, 0]

    elif isinstance(db_range, list):
        db_range = [min(db_range), max(db_range)]

    elif isinstance(db_range, tuple):
        db_range = [min(db_range), max(db_range)]

    else:
        #  default dynamic range
        print('warning: invalid image dynamic range, '
              'default dynamic range [-60, 0] is set')
        db_range = [min(db_range), max(db_range)]

    # check if 'rf' or 'iq' data on input
    is_iqdata = isinstance(rf_image[1, 1], np.complex)

    dx = x_grid[1]-x_grid[0]
    dy = y_grid[1]-y_grid[0]

    # calculate envelope
    if is_iqdata:
        amplitude_image = np.abs(rf_image)
    else:
        amplitude_image = np.abs(scs.hilbert(rf_image, axis=0))

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
