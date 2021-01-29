import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import time
import numpy as np


from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    LinSequence
)
from arrus.utils.imaging import (
    Pipeline,
    Lambda,
    Transpose,
    BandpassFilter,
    Decimation,
    QuadratureDemodulation,
    RxBeamforming,
    EnvelopeDetection,
    LogCompression,
    DynamicRangeAdjustment
)
from arrus.utils.us4r import (
    RemapToLogicalOrder
)


arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    seq = LinSequence(
        tx_aperture_center_element=np.arange(8, 183),
        tx_aperture_size=64,
        tx_focus=10e-3,
        pulse=Pulse(center_frequency=8e6, n_periods=3.5, inverse=False),
        rx_aperture_center_element=np.arange(8, 183),
        rx_aperture_size=64,
        rx_sample_range=(0, 1024),
        pri=100e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1490,
        sri=500e-3)

    def output_callback(data):
        print("Lambda!")
        return data

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=2,
        output_buffer=DataBufferSpec(type="FIFO_LOCK_FREE", n_elements=10),
        processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Transpose(axes=(0, 2, 1)),
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                RxBeamforming(),
                Lambda(function=output_callback)
            ),
            placement="/GPU:0"
        )
    )

    # Here starts communication with the device.
    session = arrus.session.Session(r"C:\Users\Public\us4r.prototxt")
    us4r = session.get_device("/Us4R:0")

    # Upload sequence on the us4r-lite device.
    buffer, const_metadata = session.upload(scheme)

    print("starting the session")
    session.start_scheme()
    time.sleep(2)
    print("stopping the session")
    session.stop_scheme()
    print("everyting OK!")


if __name__ == "__main__":
    main()