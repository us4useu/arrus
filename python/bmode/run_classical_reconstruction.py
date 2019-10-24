import numpy as np
import scipy
import scipy.signal
import argparse
import matplotlib.pyplot as plt


# Starting depth for the imaging process, [m]
START_DEPTH = 0.005

# TODO(pjarosik) replace below constants with data from the transducer's def.
# Transducer's center frequency [Hz].
CENTRAL_FREQUENCY = 5.5e6
# Transducer's pitch [m].
TRANSDUCER_PITCH = 0.00021
# Sampling frequency [Hz].
SAMPLING_FREQUENCY = 50e6


def compute_delays(n_samples, n_channels,
                   sampling_frequency, c,
                   start_depth, pitch):
    """
    Computes delay matrix for given parameters.

    :param n_samples: number of samples to consider
    :param n_channels: number of channels to consider
    :param sampling_frequency: transducer's sampling frequency [Hz]
    :param c: speed of sound [m/s]
    :param start_depth: the starting depth [m]
    :param pitch: transducer's pitch [m]
    :return: A delay matrix of shape (n_samples, n_channels)
    """
    # The distance from the line origin.
    sample_position = np.arange(0, n_samples)/sampling_frequency*c/2
    # The distance of the aperture element from the line origin.
    # The central elements should have position +/-0.5*pitch.
    element_position = (np.arange(-n_channels//2, n_channels//2)+0.5)*pitch

    # Make sample_position a column vector: (n_samples, 1).
    sample_position = sample_position.reshape((-1, 1))
    # Make element position a row vector: (1, n_channels).
    element_position = element_position.reshape((1, -1))

    # (n_samples, 1)
    tx_distance = sample_position
    # (n_samples, n_channels)
    rx_distance = np.sqrt(element_position**2 + sample_position**2)


    total_distance = tx_distance+rx_distance+start_depth
    delays = (total_distance/c)*sampling_frequency
    return delays


def delay_and_sum(data, delays):
    """
    Beamforms data using delay and sum algorithm.

    :param data: input RF data of shape (n_samples, n_channels, n_lines)
    :param delays: delay matrix of shape (n_samples, n_channels)
    :return: beamformed RF data of shape (n_samples, n_lines)
    """
    buffer = np.zeros(data.shape)
    n_samples, n_channels, n_lines = data.shape
    delays = np.maximum(delays, 0)
    delays = np.minimum(delays, n_samples-1)
    delays_floor = np.floor(delays).astype(np.int32)
    delays_ceil = np.ceil(delays).astype(np.int32)

    # Delay.
    for channel in range(n_channels):
        ch_delays_floor = delays_floor[:, channel]
        ch_delays_ceil = delays_ceil[:, channel]
        floor_coeff = delays[:, channel]-ch_delays_floor
        floor_coeff = floor_coeff.reshape((-1, 1))
        buffer[:, channel, :] = data[ch_delays_floor, channel, :]*floor_coeff \
                             + data[ch_delays_ceil, channel, :]*(1-floor_coeff)
    # Sum.
    data = np.sum(buffer, axis=1)
    return data


def filter_bandpass(data, bandwidth, axis):
    """
    Filters input data using Buttherworth filter.

    :param data: data to filter
    :param bandwidth: critical frequencies
    :param axis: axis along which the filter should be applied
    :return: filtered data
    """
    l, r = bandwidth
    b, a = scipy.signal.butter(10, l, 'highpass')
    data = scipy.signal.filtfilt(b, a, data, axis=axis)
    b, a = scipy.signal.butter(10, r, 'lowpass')
    data = scipy.signal.filtfilt(b, a, data, axis=axis)
    return data


def adjust_dynamic_range(data, max_db):
    """
    Converts to dB scale and clips to max dB.

    :param data: data to process
    :param max_db: dB threshold [dB]
    :return: processed image
    """
    nonzero_idx = data != 0
    data = 20*np.log10(np.abs(data)/np.max((np.abs(data[nonzero_idx]))))
    data = np.clip(data, -max_db, 0)
    return data


def main():
    # Read input parameters.
    parser = argparse.ArgumentParser(
        description="Performs classical reconstruction for given RF data.")
    parser.add_argument(
        "--data", dest="data",
        help="Path to the input RF data. "
             "Should have shape CSL: (Channel, Sample, Line).",
        required=True)
    parser.add_argument(
        "--speed_of_sound", dest="speed_of_sound",
        help="Speed of sound, [m/s].",
        type=float,
        required=True)
    parser.add_argument(
        "--start_depth", dest="start_depth",
        help="Start depth for the imaging process [m].",
        required=False,
        type=float,
        default=0.005
    )
    parser.add_argument(
        "--decimation_factor", dest="decimation_factor",
        help="Decimation factor.",
        required=False,
        type=int,
        default=8
    )
    parser.add_argument(
        "--max_db", dest="max_db",
        help="Dynamic range adjustment threshold.",
        required=False,
        type=float,
        default=40.0
    )
    args = parser.parse_args()

    # Load data.
    data = np.load(args.data)
    data = np.transpose(data, axes=(1, 0, 2))
    # Decimate.
    data = data[::args.decimation_factor, :, :]
    sampling_frequency = SAMPLING_FREQUENCY//args.decimation_factor
    # Compute delays.
    n_samples, n_channels, n_lines = data.shape
    delays = compute_delays(
        n_samples=n_samples,
        n_channels=n_channels,
        sampling_frequency=sampling_frequency,
        c=args.speed_of_sound,
        start_depth=args.start_depth,
        pitch=TRANSDUCER_PITCH
    )
    # Reconstruct.
    data = filter_bandpass(data, (0.05, 0.7), axis=0)
    data = delay_and_sum(data, delays)
    # Envelope detection.
    data = np.abs(scipy.signal.hilbert(data, axis=0))
    # Perform imaging.
    data = adjust_dynamic_range(data, max_db=args.max_db)
    # Display the image.
    plt.imshow(data, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
