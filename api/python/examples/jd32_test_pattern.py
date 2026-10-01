"""
AFE58JD32 (us4OEM+ 64) bring-up: acquire raw us4OEM data with the AFE ramp test
pattern on, then with normal RF data, and save both to an .npz file for analysis.

Raw data = us4OEM buffer as written by the FPGA (no remapping), one array per
buffer element: (n_firings * n_samples, 64) int16 for the JD32.
"""
import os
import threading
import time

import numpy as np

import arrus
import arrus.medium
import arrus.session
from arrus.ops.us4r import Scheme, Pulse, Tx, Rx, TxRx, TxRxSequence, DataBufferSpec

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("jd32_test_pattern.log", arrus.logging.TRACE)

# Saved next to this script (when run over \\tsclient, this is the dev machine).
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jd32_capture_all_tx.npz")
N_ELEMENTS_TO_SAVE = 2   # Buffer elements (= full sequences) per phase.
N_SAMPLES = 1024
# PRI with an odd number of microseconds on purpose: without a per-firing TX_TRIG the
# odd/even phase would flip between consecutive firings.
PRI = 201e-6
# Single-element TX sweep; RX on all elements. The TX element's own channel (TX feedthrough +
# ringing) shows up strongest, which verifies the RX channel mapping (expected: the diagonal).
TX_ELEMENTS = list(range(128))
HV_VOLTAGE = 10  # [V]
RUN_RAMP = False


class Collector:
    """Copies the first n buffer elements from the callback thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frames = []
        self.n_wanted = 0
        self.done = threading.Event()

    def arm(self, n):
        with self.lock:
            self.frames = []
            self.n_wanted = n
            self.done.clear()

    def callback(self, element):
        with self.lock:
            if len(self.frames) < self.n_wanted:
                self.frames.append(np.array(element.data, copy=True))
                if len(self.frames) == self.n_wanted:
                    self.done.set()
        element.release()


def acquire(collector, n, timeout=10.0):
    collector.arm(n)
    if not collector.done.wait(timeout):
        raise TimeoutError(f"Got only {len(collector.frames)} of {n} buffer elements.")
    return np.stack(collector.frames)


def main():
    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("us4oem.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(HV_VOLTAGE)

        n_elements = us4r.get_probe_model().n_elements
        ops = []
        for tx_element in TX_ELEMENTS:
            tx_aperture = [False] * n_elements
            tx_aperture[tx_element] = True
            ops.append(TxRx(
                Tx(aperture=tx_aperture,
                   excitation=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
                   # Explicit delay for the single active element (focus/angle breaks for 1 element).
                   delays=np.zeros(1)),
                Rx(aperture=[True] * n_elements, sample_range=(0, N_SAMPLES), downsampling_factor=1),
                pri=PRI))
        seq = TxRxSequence(ops=ops, tgc_curve=[], sri=100e-3)  # 256 firings x 201 us = 51.5 ms
        scheme = Scheme(
            tx_rx_sequence=seq,
            rx_buffer_size=4,
            output_buffer=DataBufferSpec(type="FIFO", n_elements=4),
            work_mode="HOST")
        buffer, metadata = sess.upload(scheme)

        collector = Collector()
        buffer.append_on_new_data_callback(collector.callback)

        # Phase 1: AFE ramp test pattern (optional).
        ramp = np.zeros((0,), np.int16)
        if RUN_RAMP:
            us4r.set_test_pattern("RAMP")
            sess.start_scheme()
            time.sleep(0.5)  # Let the pattern settle.
            ramp = acquire(collector, N_ELEMENTS_TO_SAVE)
            sess.stop_scheme()
            print(f"Ramp: {ramp.shape}, min {ramp.min()}, max {ramp.max()}")

        # Phase 2: normal operation (RF data).
        us4r.set_test_pattern("OFF")
        sess.start_scheme()
        time.sleep(0.5)
        rf = acquire(collector, N_ELEMENTS_TO_SAVE)
        sess.stop_scheme()
        print(f"RF: {rf.shape}, min {rf.min()}, max {rf.max()}")

        # upload() returns a list of metadata (one per sequence).
        md = metadata[0] if isinstance(metadata, (list, tuple)) else metadata
        fcm = md.data_description.custom["frame_channel_mapping"]
        results = dict(
            ramp=ramp,
            rf=rf,
            fcm_frames=np.array(fcm.frames),
            fcm_channels=np.array(fcm.channels),
            fcm_us4oems=np.array(fcm.us4oems),
            fcm_frame_offsets=np.array(fcm.frame_offsets),
            fcm_n_frames=np.array(fcm.n_frames),
            sampling_frequency=us4r.sampling_frequency,
            n_samples=N_SAMPLES,
            pri=PRI,
            tx_elements=np.asarray(TX_ELEMENTS),
            hv_voltage=HV_VOLTAGE,
            lna_gain=us4r.get_lna_gain(),
            pga_gain=us4r.get_pga_gain(),
        )
    # Write the file after the session is closed: a long write (e.g. over \\tsclient) inside the session
    # starved the host watchdog, which then stopped the us4OEM (2026-10-01).
    np.savez_compressed(OUTPUT_FILE, **results)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
