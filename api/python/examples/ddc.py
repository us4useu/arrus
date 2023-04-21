"""
This script acquires and reconstructs RF img for plane wave imaging.
GPU is required.
"""
import numpy as np
import arrus
import scipy.signal
import arrus.ops.us4r
from arrus.ops.us4r import *
from arrus.ops.imaging import PwiSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import get_bmode_imaging, get_extent
from arrus.utils.imaging import *
from collections import deque

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.TRACE)

# Here starts communication with the device.
with arrus.Session("/home/pjarosik/us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(20)

    sequence = PwiSequence(
        angles=np.array([0]),
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_sample_range=(0, 2*1024),
        speed_of_sound=1450,
        pri=400e-6,
        tgc_start=14,
        tgc_slope=2e2)

    # Imaging output grid.
    x_grid = np.arange(-15, 15, 0.1) * 1e-3
    z_grid = np.arange(5, 40, 0.1) * 1e-3

    decimation_factor = 4
    
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
            ),
            placement="/GPU:0"),
        digital_down_conversion=arrus.ops.us4r.DigitalDownConversion(
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
    np.save("ddc.npy", np.stack(q))
    print("Saved the data to ddc.npy")

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
