"""
This script acquires and reconstructs RF img for plane wave imaging.
GPU is required.
"""
import numpy as np
import arrus
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import PwiSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import get_bmode_imaging, get_extent
import sys

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
with arrus.Session(sys.argv[1]) as sess:
    us4r = sess.get_device("/Us4R:0")
    #us4r.set_hv_voltage(10)

    sequence = PwiSequence(
        angles=[0],#np.linspace(-10, 10, 11)*np.pi/180,
        pulse=Pulse(center_frequency=5e6, n_periods=2, inverse=False),
        rx_sample_range=(256, 1024*15),
        downsampling_factor=1,
        speed_of_sound=1450,
        pri=200e-6,
        tgc_start=14,
        tgc_slope=2e2)

    # Imaging output grid.
    x_grid = np.arange(-15, 15, 0.1) * 1e-3
    z_grid = np.arange(5, 100, 0.1) * 1e-3

    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid)))
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    display = Display2D(metadata=metadata, value_range=(40, 100), cmap="gray",
                        title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                        extent=get_extent(x_grid, z_grid)*1e3,
                        show_colorbar=True)
    sess.start_scheme()
    display.start(buffer)

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
