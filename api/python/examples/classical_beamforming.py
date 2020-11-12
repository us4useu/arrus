import arrus.session
import numpy as np
import matplotlib.pyplot as plt
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse
import arrus.logging
import arrus.utils.us4r
import time

arrus.logging.set_clog_level(arrus.logging.INFO)
arrus.logging.add_log_file("test.log", arrus.logging.TRACE)

session = arrus.session.Session(
    r"C:\Users\pjarosik\src\x-files\customers\nanoecho\nanoecho_magprobe_002.prototxt")

seq = LinSequence(
    tx_aperture_center_element=np.arange(8, 182),
    tx_aperture_size=64,
    tx_focus=30e-3,
    pulse=Pulse(center_frequency=5e6, n_periods=3.5, inverse=False),
    rx_aperture_center_element=np.arange(8, 182),
    rx_aperture_size=64,
    rx_sample_range=(0, 4096),
    pri=100e-6,
    downsampling_factor=1,
    tgc_start=14,
    tgc_slope=2e2,
    speed_of_sound=1490)

us4r = session.get_device("/Us4R:0")
us4r.set_voltage(30)
buffer = us4r.upload(seq, mode="sync")

print("Starting the device.")


def display_data(data):
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.imshow(data)
    ax.set_aspect('auto')
    fig.show()


us4r.start()

times = []

print("Starting")

for i in range(100):
    print(i)
    start = time.time()
    data, metadata = buffer.tail()
    buffer.release_tail()
    times.append(time.time()-start)

print("Done")
print(f"Average time: {np.mean(times)}")

us4r.stop()

# remap_step = arrus.utils.us4r.RemapToLogicalOrder()
# remap_step.set_pkgs(num_pkg=np)
# remapped_data, metadata = remap_step(data, metadata)

# print("Saving data")
# np.save("test.npy", data)
# print("Data saved")
#
# buffer.release_tail()
#
# print("Stopping the device.")
# us4r.stop()
