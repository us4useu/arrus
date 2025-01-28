"""
This script shows how users can run their own custom callback when new data
is acquired to the PC memory.

A callback measuring acquisition frame is used. The actual frame rate
can be controlled using PwiSequence sri parameter.
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import time
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import LinSequence


arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


class Timer:
    def __init__(self):
        self.last_time = time.time()

    def callback(self, element):
        now = time.time()
        dt = now-self.last_time
        print(f"Delta t: {dt:6.2f}, data size: {element.data.nbytes} bytes")
        print(f"Frame: {element.data.shape}, min: {np.min(element.data)}, max: {np.max(np.max(element.data))}")
        self.last_time = now
        element.release()


def main():
    seq = LinSequence(
        tx_aperture_center_element=np.arange(8, 183),
        tx_aperture_size=64,
        tx_focus=24e-3,
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_aperture_center_element=np.arange(8, 183),
        rx_aperture_size=64,
        rx_sample_range=(0, 1024),
        pri=200e-6,
        tgc_start=14,
        tgc_slope=2e2,
        downsampling_factor=2,
        speed_of_sound=1450)

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=4,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=4),
        work_mode="HOST")

    # Here starts communication with the device.
    with arrus.Session("us4r_L3-9i-D.prototxt") as sess:
        ultrasound = sess.get_device("/Ultrasound:0")
        # us4r.set_hv_voltage(10)
        # Upload sequence on the us4r-lite device.
        buffer, const_metadata = sess.upload(scheme)
        timer = Timer()
        buffer.append_on_new_data_callback(timer.callback)
        sess.start_scheme()
        work_time = 10  # [s]
        print(f"Running for {work_time} s")
        time.sleep(10)
    # When we exit the above scope, the session and scheme is properly closed.
    print("Finished.")


if __name__ == "__main__":
    main()