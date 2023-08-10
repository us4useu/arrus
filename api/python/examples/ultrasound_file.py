import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import queue
import numpy as np

from arrus.ops.us4r import (
    Scheme,
    Pulse,
)
from arrus.ops.imaging import (
    LinSequence
)
from arrus.utils.imaging import (
    Pipeline,
    SelectFrames,
    Squeeze,
)
from arrus.utils.gui import (
    Display2D
)

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


def main():
    with arrus.Session("/home/pjarosik/tmp/test.prototxt") as sess:
        ultrasound = sess.get_device("/Ultrasound:0")
        # NOTE: file device does not allow to set voltage, etc.

        sequence = LinSequence(
            tx_aperture_center_element=np.arange(8, 183),
            tx_aperture_size=64,
            tx_focus=24e-3,
            pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
            rx_aperture_center_element=np.arange(8, 183),
            rx_aperture_size=64,
            rx_sample_range=(0, 2048),
            pri=200e-6,
            tgc_start=14,
            tgc_slope=2e2,
            speed_of_sound=1450
        )

        scheme = Scheme(
            tx_rx_sequence=sequence,
            processing=Pipeline(
                steps=(
                    Squeeze(),
                    SelectFrames([0]),
                    Squeeze()
                ),
                placement="/GPU:0"
            )
        )
        buffer, metadata = sess.upload(scheme)
        display = Display2D(metadata=metadata, value_range=(0, 10))
        sess.start_scheme()
        display.start(buffer)
        print("Display closed, stopping the script.")


if __name__ == "__main__":
    main()
