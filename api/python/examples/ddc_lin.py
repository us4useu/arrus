"""
This script acquires and reconstructs RF img for plane wave imaging.
GPU is required.
"""
import numpy as np
import arrus
import scipy.signal
from arrus.ops.us4r import *
from arrus.ops.imaging import *
from arrus.utils.gui import Display2D
from arrus.utils.imaging import *
from collections import deque

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
with arrus.Session("/home/pjarosik/us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(20)

    sequence = LinSequence(
        tx_aperture_center_element=np.arange(0, 192),
        tx_aperture_size=64,
        tx_focus=24e-3,
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_aperture_center_element=np.arange(0, 192),
        rx_aperture_size=64,
        rx_sample_range=(0, 2048),
        pri=200e-6,
        tgc_start=14,
        tgc_slope=2e2,
        speed_of_sound=1450)

    # Imaging output grid.
    x_grid = np.arange(-15, 15, 0.1) * 1e-3
    z_grid = np.arange(5, 40, 0.1) * 1e-3

    decimation_factor = 2
    
    if np.modf(decimation_factor)[0] == 0.25 or np.modf(decimation_factor)[0] == 0.75 :
        filter_order = np.uint32(decimation_factor*64)
    elif np.modf(decimation_factor)[0] == 0.5 :
        filter_order = np.uint32(decimation_factor*32)
    else :
        filter_order = np.uint32(decimation_factor*16)

    cutoff = 6e6
    fs = us4r.sampling_frequency
    coeffs = scipy.signal.firwin(filter_order, cutoff, fs=fs)
    coeffs = coeffs[filter_order//2:]
    
    q = deque(maxlen=10)
    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=Pipeline(
            steps=(
                RemapToLogicalOrderV2(),
                # Save the data to q.
                Lambda(lambda data: (q.append(data.get()), data)[1]),
                ToRealOrComplex(),
                EnvelopeDetection(),
                Squeeze(),
                # Display the center of probe's aperture.
                SelectFrame([96]),
                Squeeze()
            ),
            placement="/GPU:0"),
        digital_down_conversion=DigitalDownConversion(
            demodulation_frequency=6e6,
           decimation_factor=decimation_factor,
           fir_coefficients=coeffs)
    )
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    display = Display2D(metadata=metadata, value_range=(-1000, 1000),
                        show_colorbar=True)
    sess.start_scheme()
    display.start(buffer)
    np.save("ddc_lin.npy", np.stack(q))
    print("Saved the data to ddc_lin.npy")

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
