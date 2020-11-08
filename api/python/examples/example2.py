import arrus.session
import numpy as np
import matplotlib.pyplot as plt
from arrus.ops.us4r import (
    TxRxSequence,
    TxRx,
    Tx,
    Rx,
    Pulse
)
import arrus.logging
import arrus.utils.us4r

arrus.logging.set_clog_level(arrus.logging.TRACE)
arrus.logging.add_log_file("test.log", arrus.logging.TRACE)

session = arrus.session.Session(
    r"C:\Users\pjarosik\src\x-files\customers\nanoecho\nanoecho_magprobe_002.prototxt")

rx_aperture = np.zeros((192,), dtype=np.bool)
rx_aperture[:192] = True
seq = TxRxSequence(
    ops=[
        TxRx(
            tx=Tx(
                aperture=np.ones((192, ), dtype=np.bool),
                delays=(np.arange(0, 192)*1e-8).astype(np.float32),
                excitation=Pulse(
                    center_frequency=5e6,
                    n_periods=3.5,
                    inverse=True
                )
            ),
            rx=Rx(
                aperture=rx_aperture,
                sample_range=(0, 4096),
                downsampling_factor=1
            ),
            pri=1000e-6),
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
                aperture=rx_aperture,
                sample_range=(0, 4096),
                downsampling_factor=1
            ),
            pri=1000e-6),
        TxRx(
            tx=Tx(
                aperture=np.ones((192, ), dtype=np.bool),
                delays=np.flip((np.arange(0, 192)*1e-8).astype(np.float32)),
                excitation=Pulse(
                    center_frequency=5e6,
                    n_periods=3.5,
                    inverse=True
                )
            ),
            rx=Rx(
                aperture=rx_aperture,
                sample_range=(0, 4096),
                downsampling_factor=1
            ),
            pri=1000e-6),
    ],
    tgc_curve=np.array([])
)

us4r = session.get_device("/Us4R:0")
us4r.set_voltage(30)
buffer = us4r.upload(seq)

print("Starting the device.")


def display_data(data, frame):
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.imshow(data[frame, :, :])
    ax.set_aspect('auto')
    fig.show()


def display_data(data):
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.imshow(data)
    ax.set_aspect('auto')
    fig.show()


us4r.start()
data, metadata = buffer.tail()
remap_step = arrus.utils.us4r.RemapToLogicalOrder()
remap_step.set_pkgs(num_pkg=np)
remapped_data, metadata = remap_step(data, metadata)

# print("Saving data")
# np.save("test.npy", data)
# print("Data saved")
#
# buffer.release_tail()
#
# print("Stopping the device.")
# us4r.stop()
