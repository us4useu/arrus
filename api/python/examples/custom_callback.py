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
from arrus.ops.imaging import PwiSequence


arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)


class Timer:
    def __init__(self):
        self.last_time = time.time()

    def callback(self, element):
        now = time.time()
        dt = now-self.last_time
        print(f"Delta t: {dt:6.2f}, data size: {element.data.nbytes} bytes",
              end="\r")
        self.last_time = now
        element.release()


def main():
    seq = PwiSequence(
        angles=np.asarray([-5, 0, 5])*np.pi/180,
        pulse=Pulse(center_frequency=8e6, n_periods=3, inverse=False),
        rx_sample_range=(0, 4096),
        downsampling_factor=1,
        speed_of_sound=1490,
        pri=200e-6,
        sri=10e-3,
        tgc_start=14,
        tgc_slope=2e2)

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=4,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=12),
        work_mode="ASYNC")

    # Here starts communication with the device.
    with arrus.Session() as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(10)
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