import numpy as np
import arrus
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import PwiSequence, LinSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import get_bmode_imaging, get_extent

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("swe.log", arrus.logging.INFO)

with arrus.Session("us4r_L3-9i-D.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(15)

    sequence = PwiSequence(
            angles=np.linspace(-4, -2, 2, 4, 64)*np.pi/180,
            pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
            rx_sample_range=(0, 1024*4),
            downsampling_factor=1,
            speed_of_sound=1540,
            pri=200e-6,
            tgc_start=14,
            tgc_slope=2e2)

# Imaging output grid.
    x_grid = np.arange(-15, 15, 0.2) * 1e-3
    z_grid = np.arange(5, 45, 0.2) * 1e-3

    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid)),
        constants=tx_focuses
    )
    
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    display = Display2D(metadata=metadata, value_range=(20, 80), cmap="gray",
                        title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                        extent=get_extent(x_grid, z_grid)*1e3,
                        show_colorbar=True)
    sess.start_scheme()
    sess.set("/Us4R:0/sequence/txFocus", tx_focuses[10])
    display.start(buffer)

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
