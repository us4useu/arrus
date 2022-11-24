"""
This script acquires and reconstructs RF img for plane wave imaging.
GPU is required.
"""
import numpy as np
import arrus
from arrus.ops.us4r import Scheme, Pulse
from arrus.ops.imaging import PwiSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import *
from collections import deque
import scipy.io
from datetime import datetime

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
with arrus.Session("./us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(50)

    sequence = PwiSequence(
        angles=np.array([0]*150), # np.linspace(-10, 10, 150)*np.pi/180,
        pulse=Pulse(center_frequency=5e6, n_periods=2, inverse=False),
        rx_sample_range=(0, 4224),
        downsampling_factor=1,
        speed_of_sound=1540,
        pri=150e-6,
        tgc_curve=[],
        n_repeats=1
    )

    # Imaging output grid.
    x_grid = np.arange(-20, 20, 0.1) * 1e-3
    z_grid = np.arange(5, 50, 0.1) * 1e-3

    rf_queue = deque(maxlen=2)
    bmode_que = deque(maxlen=2)

    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=Pipeline(
            steps=(
                # Channel data pre-processing.
                RemapToLogicalOrder(),
                Lambda(lambda data: (rf_queue.append(data.get()), data)[1]),
                SelectSequence([0]),
                Squeeze(),
                Transpose(axes=(0, 2, 1)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                # Data beamforming.
                ReconstructLri(x_grid=x_grid, z_grid=z_grid),
                Lambda(lambda data: (bmode_que.append(data.get()), data)[1]),
                Mean(axis=0),
                # Post-processing to B-mode image.
                EnvelopeDetection(),
                Transpose(),
                LogCompression(),
            ),
            placement="/GPU:0")

    )
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    display = Display2D(metadata=metadata, value_range=(20, 80), cmap="gray",
                        title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
                        extent=get_extent(x_grid, z_grid)*1e3,
                        show_colorbar=True)
    sess.start_scheme()
    display.start(buffer)
    scipy.io.savemat(f"{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.mat", dict(rf=np.stack(rf_queue), bmode=np.stack(bmode_que)))

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
