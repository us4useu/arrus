"""
This script acquires and reconstructs RF img for classical imaging scheme
(scanline by scanline).

GPU usage is recommended.
"""
import numpy as np
import arrus
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import LinSequence
from arrus.utils.gui import Display2D
from arrus.utils.imaging import *

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


last_timestamp = 0.0

# Here starts communication with the device.
with arrus.Session("/home/pjarosik/us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(5)

    sequence = LinSequence(
        tx_aperture_center_element=np.arange(8, 183),
        tx_aperture_size=64,
        tx_focus=10e-3,
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_aperture_center_element=np.arange(8, 183),
        rx_aperture_size=64,
        rx_sample_range=(0, 1024),
        pri=100e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1450,
        sri=17.5e-3,
        )

    # Imaging output grid.
    x_grid = np.arange(-15, 15, 0.2) * 1e-3
    z_grid = np.arange(5, 45, 0.2) * 1e-3


    def lambda_func(data):
        global last_timestamp
        t = data[0, 4:8].get().view(np.uint64).item()/65e6
        print(t-last_timestamp)
        last_timestamp = t
        return data

    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=Pipeline(
            steps=(
                # Channel data pre-processing.
                Lambda(lambda_func),
                RemapToLogicalOrder(),
                Transpose(axes=(0, 1, 3, 2)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4,
                           cic_order=2),
                # # Data beamforming.
                RxBeamforming(),
                # # Post-processing to B-mode image.
                EnvelopeDetection(),
                Transpose(axes=(0, 2, 1)),
                ScanConversion(x_grid, z_grid),
                LogCompression(),
                Squeeze()
            ),
            placement="/GPU:0"),
        # get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid)),
        rx_buffer_size=15,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=15),
        work_mode="ASYNC"
    )
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    # display = Display2D(metadata=metadata, value_range=(20, 80), cmap="gray",
    #                     title="B-mode", xlabel="OX (mm)", ylabel="OZ (mm)",
    #                     extent=get_extent(x_grid, z_grid)*1e3,
     #                    show_colorbar=True)
    sess.start_scheme()
    []
    for i in range(100):
        frame = buffer.get()[0]

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
