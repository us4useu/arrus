"""
This script acquires and reconstructs RF img for plane wave imaging.
GPU is required.
"""
import numpy as np
import arrus
import scipy.signal
from arrus.ops.us4r import *
from arrus.ops.imaging import PwiSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import get_bmode_imaging, get_extent
from arrus.utils.imaging import *
import matplotlib.pyplot as plt

arrus.set_clog_level(arrus.logging.TRACE)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
with arrus.Session("C:/Users/Public/us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(20)

    sequence = PwiSequence(
        angles=np.array([0]),
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_sample_range=(0, 2*1024),
        downsampling_factor=1,
        speed_of_sound=1450,
        pri=400e-6,
        tgc_start=14,
        tgc_slope=2e2)

    # Imaging output grid.
    x_grid = np.arange(-15, 15, 0.1) * 1e-3
    z_grid = np.arange(5, 35, 0.1) * 1e-3

    decimation_factor = 4
    filter_order = decimation_factor*16
    cutoff = 6e6
    fs = us4r.sampling_frequency
    coeffs = scipy.signal.firwin(filter_order, cutoff/(fs/2))
    # coeffs = np.ones(filter_order)
    coeffs = coeffs[filter_order//2:]
    print(coeffs.shape)
    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=Pipeline(
            steps=(
                # Channel data pre-processing.
                Lambda(lambda data: data),
            ),
            placement="/GPU:0"),
        work_mode="MANUAL",
        digital_down_conversion=DigitalDownConversion(
            demodulation_frequency=6e6,
           decimation_factor=decimation_factor,
           fir_coefficients=coeffs)
    )
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)

    sess.run()
    data = buffer.get()[0]

    sess.run()
    data = buffer.get()[0]

    sess.run()
    data = buffer.get()[0]

    data = data.reshape(2, 3, 2048, 2, 32)
    # data = data.reshape((4096, 2, 32))

    # plt.imshow(data[0, 0, :, 0, :])
    # plt.show()
    #
    # plt.imshow(data[0, 0, :, 1, :])
    # plt.show()
    #
    # plt.imshow(np.abs(data[:, :, 0]+1j*data[:, :, 1]))
    # plt.show()

    img = np.ones((2048, 2, 192), dtype=np.int16)
    data = data.reshape(2, 3, 2048, 2, 32)
    for rx in range(3):
        for u in range(2):
            subap = rx*64 + u*32
            img[:, :, subap:(subap+32)] = data[u, rx, :, :, :]
    cplx = img[:, 0, :] + 1j*img[:, 1, :]
    plt.imshow(np.real(cplx), vmin=-1000, vmax=1000)
    plt.show()
    plt.imshow(np.imag(cplx), vmin=-1000, vmax=1000)
    plt.show()
    envelope = np.abs(cplx)
    plt.imshow(envelope[200:, :], vmin=-2000, vmax=2000)
    plt.show()

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
