"""
This script shows how to perform a mix of phased-array with moving aperture scanning.

The sequence contains: TX with negative angles (to extend the FOV from left),
a sequence of moving aperture, the TX with positive angles.

GPU is required.
"""
import numpy as np
import arrus
from arrus.ops.us4r import *
from arrus.utils.gui import Display2D
from arrus.utils.imaging import *
from collections import deque

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
c = 1450
with arrus.Session("/home/pjarosik/us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(40)

    n_elements = us4r.get_probe_model().n_elements

    focus = 20e-3
    pulse = Pulse(center_frequency=6e6, n_periods=2, inverse=False)
    aperture_size = 64
    center_element_left = aperture_size//2-1
    center_element_right = n_elements-aperture_size//2

    left_aperture = Aperture(center_element=center_element_left, size=aperture_size)
    right_aperture = Aperture(center_element=center_element_right, size=aperture_size)

    sample_range = (0, 4*1024)
    left_fov_ops = [
        TxRx(
            Tx(aperture=left_aperture,
               excitation=pulse,
               angle=a/180*np.pi, speed_of_sound=c, focus=focus),
            Rx(aperture=left_aperture, sample_range=sample_range),
            pri=200e-6
        )
        for a in range(-30, 0)
    ]
    center_fov_ops = [
        TxRx(
            Tx(aperture=Aperture(center_element=i, size=aperture_size),
               excitation=pulse,
               angle=0, speed_of_sound=c, focus=focus),
            Rx(aperture=Aperture(center_element=i, size=aperture_size),
               sample_range=sample_range),
            pri=200e-6
        )
        for i in range(center_element_left, center_element_right+1)
    ]
    right_fov_ops = [
        TxRx(
            Tx(aperture=right_aperture,
               excitation=pulse,
               angle=a/180*np.pi, speed_of_sound=c, focus=focus),
            Rx(aperture=right_aperture, sample_range=sample_range),
            pri=200e-6
        )
        for a in range(0, 30)
    ]

    sequence = TxRxSequence(
        ops=left_fov_ops+center_fov_ops+right_fov_ops,
        tgc_curve=[]
    )

    rf_queue = deque(maxlen=3)
    metadata_queue = deque(maxlen=1)

    def acquire(data):
        rf_queue.append(data.get())
        return data

    def acquire_metadata(m):
        metadata_queue.append(m)
        return m

    # Imaging output grid.
    x_grid = np.arange(-30, 30, 0.1) * 1e-3
    z_grid = np.arange(5, 45, 0.1) * 1e-3

    imaging = Pipeline(
        steps=(
            RemapToLogicalOrder(),
            Lambda(acquire, acquire_metadata),
            Transpose(axes=(0, 1, 3, 2)),
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=4, cic_order=2),
            RxBeamforming(),
            EnvelopeDetection(),
            Transpose(axes=(0, 2, 1)),
            ScanConversion(x_grid, z_grid),
            Mean(axis=0),
            LogCompression(),
        ),
        placement="/GPU:0"
    )



    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=imaging,
    )
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    display = Display2D(metadata=metadata, value_range=(20, 80), cmap="gray",
                        title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                        extent=get_extent(x_grid, z_grid)*1e3,
                        show_colorbar=True)
    sess.start_scheme()
    display.start(buffer)

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
