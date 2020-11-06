import arrus.session
import numpy as np
from arrus.ops.us4r import (
    TxRxSequence,
    TxRx,
    Tx,
    Rx,
    Pulse
)
import arrus.logging

arrus.logging.set_clog_level(arrus.logging.TRACE)
arrus.logging.add_log_file("test.log", arrus.logging.TRACE)

session = arrus.session.Session(
    r"C:\Users\pjarosik\src\x-files\customers\nanoecho\nanoecho_magprobe_002.prototxt")

seq = TxRxSequence(
    operations=[
        TxRx(
            tx=Tx(
                aperture=np.ones((192, ), dtype=np.bool),
                delays=np.zeros((192, ), dtype=np.float32),
                excitation=Pulse(
                    center_frequency=5e6,
                    n_periods=3.5,
                    inverse=True
                )
            ),
            rx=Rx(
                aperture=np.ones((192, ), dtype=np.bool),
                sample_range=(0, 4096),
                downsampling_factor=1
            ),
            pri=1000e-6)
    ],
    tgc_curve=np.array([])
)

us4r = session.get_device("/Us4R:0")
us4r.set_voltage(30)
buffer = us4r.upload(seq)

print("Starting the device.")
us4r.start()

# data, metadata = buffer.tail()
# print("Saving data")
# np.save("test.npy", data)
# print("Data saved")
#
# buffer.release_tail()
#
# print("Stopping the device.")
# us4r.stop()
