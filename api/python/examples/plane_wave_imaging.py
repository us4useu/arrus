"""
This script acquires and reconstructs RF img for plane wave imaging.
GPU is required.
"""
import numpy as np
import arrus
import arrus.session
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import PwiSequence
from arrus.utils.imaging import get_bmode_imaging, get_extent
from arrus.utils.gui import Display2D
from collections import deque
import time
from api.python.examples.custom_callback import Timer  # Importer la classe Timer depuis custom_callback.py

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
with arrus.Session("us4r_L3-9i-D.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(20)

    sequence = PwiSequence(
        angles=np.linspace(-3, 3, 3)*np.pi/180,
        # angles = np.array([-4, -2,  2, 4]) * np.pi / 180,
        pulse=Pulse(center_frequency=5e6, n_periods=3, inverse=False),
        # rx_sample_range=(0, 1024*3),
        rx_depth_range=(0, 0.03),
        downsampling_factor=1,
        speed_of_sound=1540,
        pri=83.3e-6,
        tgc_start=1,
        tgc_slope=2e2,
        n_repeats = 50)

    # Imaging output grid.
    x_grid = np.arange(-20, 20, 0.1) * 1e-3
    z_grid = np.arange(5, 35, 0.1) * 1e-3   
    scheme = Scheme(
        
        tx_rx_sequence=sequence,
        processing=get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid))
    )
        
    # Upload sequence on the us4r-lite device.
    buffer, const_metadata = sess.upload(scheme)
    timer = Timer()
    buffer.append_on_new_data_callback(timer.callback)

    display = Display2D(metadata=const_metadata, value_range=(20, 80), cmap="gray",
                        title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                        extent=get_extent(x_grid, z_grid)*1e3, 
                        show_colorbar=True)
        
    sess.start_scheme()
    work_time = 10  # [s]
    print(f"Running for {work_time} s")
    time.sleep(10)
    
    # display.start(buffer)


# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
