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
import sys
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec, TxRxSequence, TxRx, Tx, Rx
from arrus.ops.imaging import LinSequence


arrus.set_clog_level(arrus.logging.TRACE)
arrus.add_log_file("test.log", arrus.logging.TRACE)

class Timer:
    def __init__(self):
        pass

    def callback(self, element):
        element.release()

def main(n_tx_rx = 10, test_buf_multiplier = 1):
    print(f"DMA PERF TEST called with n_tx_rx = {n_tx_rx}, test_buf_multiplier = {test_buf_multiplier}")

    # Here starts communication with the device.
    with arrus.Session("/home/mila/work/setup-olympus.prototxt") as sess:
        ultrasound = sess.get_device("/Us4R:0")
        n_elements = ultrasound.get_probe_model().n_elements
        print(f"DMA PERF TEST n_elements = {n_elements}")

        seq = TxRxSequence(
                    ops=[
                        TxRx(
                            Tx(aperture=[False]*n_elements,
                                excitation=Pulse(center_frequency=6e6, n_periods=2,
                                                    inverse=False),
                                # Custom delays 1.
                                delays=[]),
                            Rx(aperture=[True]*n_elements,
                                sample_range=(0, 65472-(64*20)),
                                downsampling_factor=1),
                            pri=1e-3
                        ),
                    ]*n_tx_rx,
                    # Turn off TGC.
                    tgc_curve=[]#,  # [dB]
                    # Time between consecutive acquisitions, i.e. 1/frame rate.
            )
        
        scheme = Scheme(
            tx_rx_sequence=seq,
            rx_buffer_size=4,
            output_buffer=DataBufferSpec(type="FIFO", n_elements=4),
            work_mode="ASYNC"
            )

        seqdma_buffer_size = 512 * n_tx_rx * test_buf_multiplier
        print(f"DMA PERF TEST SeqDMA buffer size: {seqdma_buffer_size}")
        ultrasound.set_seq_dma_buffer_size(seqdma_buffer_size)
        ultrasound.set_hv_voltage(5)
        ultrasound.set_stop_on_overflow(False)
        # Upload sequence on the us4r-lite device.
        buffer, const_metadata = sess.upload(scheme)
        timer = Timer()
        buffer.append_on_new_data_callback(timer.callback)
        sess.start_scheme()

        print("Running for 1 minute")

        time.sleep(60)
    # When we exit the above scope, the session and scheme is properly closed.
    print("Finished.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python custom_callback.py <n_tx_rx> <test_buf_multiplier>")
        sys.exit(1)
    main(int(sys.argv[1]), int(sys.argv[2]))
