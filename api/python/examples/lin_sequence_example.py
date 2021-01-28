import arrus
import arrus.session

import numpy as np

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    LinSequence
)


def main():
    seq = LinSequence(
        tx_aperture_center_element=np.arange(8, 183),
        tx_aperture_size=64,
        tx_focus=30e-3,
        pulse=Pulse(center_frequency=8e6, n_periods=3.5, inverse=False),
        rx_aperture_center_element=np.arange(8, 183),
        rx_aperture_size=64,
        rx_sample_range=(0, 2048),
        pri=200e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1490,
        sri=500e-3)

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=2,
        output_buffer=DataBufferSpec(type="FIFO_LOCK_FREE", n_elements=10)
    )

    # Here starts communication with the device.
    session = arrus.session.Session(r"C:\Users\Public\us4r.prototxt")
    us4r = session.get_device("/Us4R:0")

    # Upload sequence on the us4r-lite device.
    buffer, const_metadata = session.upload(scheme)
    print("Everyting OK!")


if __name__ == "__main__":
    main()