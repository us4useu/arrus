import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import queue

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    PwiSequence
)
from arrus.utils.imaging import (
    Pipeline,
    Lambda,
    Transpose,
    BandpassFilter,
    Decimation,
    QuadratureDemodulation,
    RxBeamformingImg,
    EnvelopeDetection,
    LogCompression,
    DynamicRangeAdjustment,
    Enqueue,
    SelectFrames
)
from arrus.utils.us4r import (
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D
)


arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():

    seq = PwiSequence(
        angles=np.asarray([-5, 0, 5])*np.pi/180,
        pulse=Pulse(center_frequency=8e6, n_periods=3, inverse=False),
        rx_sample_range=(0, 4096),
        downsampling_factor=2,
        speed_of_sound=1490,
        pri=100e-6,
        sri=20e-3,
        tgc_start=14,
        tgc_slope=0)

    display_input_queue = queue.Queue(1)

    x_grid = np.linspace(-15, 15, 256) * 1e-3
    z_grid = np.linspace(0, 40, 256) * 1e-3

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=4,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=12),
        work_mode="ASYNC",
        processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Transpose(axes=(0, 2, 1)),
                SelectFrames(frames=[0]),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                RxBeamformingImg(x_grid=x_grid, z_grid=z_grid),
                EnvelopeDetection(),
                Transpose(),
                LogCompression(),
                Enqueue(display_input_queue, block=False, ignore_full=True)
            ),
            placement="/GPU:0"
        )
    )

    # Here starts communication with the device.
    with arrus.Session(r"C:\Users\Public\us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(50)

        # Upload sequence on the us4r-lite device.
        buffer, const_metadata = sess.upload(scheme)
        display = Display2D(const_metadata, value_range=(20, 80), cmap="gray")
        sess.start_scheme()
        display.start(display_input_queue)
        print("Display closed, stopping the script.")

    # When we exit the above scope, the session and scheme is properly closed.
    print("Stopping the example.")


if __name__ == "__main__":
    main()