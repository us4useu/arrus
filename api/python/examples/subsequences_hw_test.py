"""Hardware test of the session.set_subsequences method (an arbitrary list of TX/RXs).

Run with: python subsequences_hw_test.py
Adjust CFG/VOLTAGE below to your system; the test was developed on a us4R with 2 us4OEMs and
an olympus 10l128 probe (64 elements), TX voltage 10 V, single-element TX apertures.

Two independent checks of "did the hardware really run exactly the selected TX/RXs?":

1. CONTENT: TX/RX i transmits on the probe element ELEMENT_STEP*i only, so the RX channel
   profile of the TX crosstalk is a signature of the TX/RX that produced a given frame. The
   signatures are acquired once for the full sequence and then matched against.
2. TIMING: TX/RX i has its own PRI (PRI0 + i*PRI_STEP), so the duration of one sequence
   repetition is a unique fingerprint of the set of TX/RXs the sequencer executed.

The same `processing` object is passed to every set_subsequences call, so the imaging
pipeline goes through the update path (instead of being re-created).
"""
import os
import sys
import time
import numpy as np

import arrus
import arrus.session
import arrus.logging
from arrus.ops.us4r import Scheme, Pulse, Tx, Rx, TxRx, TxRxSequence, DataBufferSpec
from arrus.utils.imaging import Pipeline, Processing, RemapToLogicalOrder

arrus.set_clog_level(arrus.logging.ERROR)

# The session configuration; can be overridden with the ARRUS_CFG environment variable.
CFG = os.environ.get("ARRUS_CFG", "/opt/us4us/us4ndt64.prototxt")
VOLTAGE = 10
N_OPS = 16
ELEMENT_STEP = 4           # TX/RX i transmits on the element ELEMENT_STEP*i
N_SAMPLES = 1024
PRI0 = 2000e-6
PRI_STEP = 200e-6
N_FRAMES = 12              # frames to average the frame rate over
# The crosstalk is searched for in the first samples (the TX ringdown).
CROSSTALK_SAMPLES = 300

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"    {'OK  ' if condition else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")


def get_sequence(n_elements):
    ops = []
    for i in range(N_OPS):
        tx_aperture = np.zeros(n_elements, dtype=bool)
        tx_aperture[(ELEMENT_STEP*i) % n_elements] = True
        ops.append(
            TxRx(
                Tx(aperture=tx_aperture,
                   excitation=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
                   delays=[0.0]),
                Rx(aperture=[True]*n_elements, sample_range=(0, N_SAMPLES),
                   downsampling_factor=1),
                pri=PRI0 + i*PRI_STEP,
            )
        )
    return TxRxSequence(ops=ops, tgc_curve=[24.0]*16)


def expected_duration(ops):
    return sum(PRI0 + i*PRI_STEP for i in ops)


class Acquisition:
    """Collects the beamformer-free (remapped) frames together with their arrival times."""

    def __init__(self):
        self.data = []
        self.times = []

    def __call__(self, element):
        t = time.time()
        array = element.arrays[0].copy()
        element.release()
        self.times.append(t)
        self.data.append(array)

    def collect(self, n, timeout=15.0):
        self.data, self.times = [], []
        t0 = time.time()
        while len(self.data) < n:
            if time.time()-t0 > timeout:
                raise TimeoutError(f"Got only {len(self.data)} frames in {timeout} s.")
            time.sleep(0.005)
        # Average the repetitions (dropping the warm-up ones) -- a single acquisition of the
        # TX crosstalk is noisy.
        frames = np.stack([np.asarray(d, dtype=np.float32) for d in self.data[2:n]], axis=0)
        return np.mean(np.abs(frames), axis=0), float(np.mean(np.diff(np.asarray(self.times[2:n]))))


def rx_profiles(frame):
    """Per-frame RX channel profile: the max TX crosstalk amplitude on each RX channel."""
    # frame: (n_frames, n_samples, n_channels)
    profiles = np.max(np.asarray(frame, dtype=np.float32)[:, :CROSSTALK_SAMPLES, :], axis=1)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    return profiles/np.maximum(norms, 1e-9)


def identify_ops(frame, reference):
    """Identifies the TX/RX each acquired frame comes from, by matching the RX channel profile
    against the reference profiles acquired for the full sequence."""
    return np.argmax(rx_profiles(frame) @ reference.T, axis=1)


def verify(name, acq, ops, reference=None):
    print(f"[{name}] ops={list(ops)}")
    frame, interval = acq.collect(N_FRAMES)
    if frame.ndim == 4:  # (batch, n_frames, n_samples, n_channels)
        frame = frame[0]
    expected_interval = expected_duration(ops)

    check(f"{name}: number of frames", frame.shape[0] == len(ops),
          f"{frame.shape[0]} (expected {len(ops)}), full shape {frame.shape}")
    if reference is not None and frame.shape[0] == len(ops):
        detected = identify_ops(frame, reference)
        check(f"{name}: the TX/RXs identified in the data",
              np.array_equal(detected, np.asarray(ops)),
              f"identified {list(detected)}")
    check(f"{name}: sequence duration",
          abs(interval-expected_interval)/expected_interval < 0.05,
          f"{interval*1e6:.1f} us (expected {expected_interval*1e6:.1f} us)")
    return frame


def main():
    acq = Acquisition()
    pipeline = Pipeline(steps=(RemapToLogicalOrder(), ), placement="/GPU:0")
    processing = Processing(graph=pipeline, callback=acq)

    with arrus.Session(CFG) as sess:
        us4r = sess.get_device("/Us4R:0")
        n_elements = us4r.get_probe_model().n_elements
        print(f"Probe: {n_elements} elements, TX voltage: {VOLTAGE} V")
        us4r.set_hv_voltage(VOLTAGE)
        try:
            scheme = Scheme(
                tx_rx_sequence=get_sequence(n_elements),
                rx_buffer_size=4,
                output_buffer=DataBufferSpec(type="FIFO", n_elements=8),
                work_mode="ASYNC",
                processing=processing,
            )
            sess.upload(scheme)
            us4r.set_stop_on_overflow(False)
            sess.start_scheme()

            # 1. The full sequence: also the reference RX profile of each TX/RX.
            full_frame = verify("full sequence", acq, list(range(N_OPS)))
            reference = rx_profiles(full_frame)

            def set_subsequence(ops):
                """set_subsequences requires the scheme to be stopped."""
                sess.stop_scheme()
                t = time.time()
                if isinstance(ops, slice):
                    sess.set_subsequences([ops], processing=processing)
                else:
                    sess.set_subsequences([list(ops)], processing=processing)
                dt = time.time()-t
                sess.start_scheme()
                return dt

            # 2. Non-consecutive sub-sequences.
            for ops in ([2, 3, 5, 8, 13], [0, 1, 15], [7], [1, 4, 9], [0, 2, 4, 6, 8, 10, 12, 14]):
                dt = set_subsequence(ops)
                print(f"  set_subsequences([{ops}]): {dt*1e3:.1f} ms")
                verify(f"ops {ops}", acq, ops, reference)

            # 3. A contiguous sub-sequence (the classic slice API).
            set_subsequence(slice(4, 9))
            verify("slice(4, 9)", acq, list(range(4, 9)), reference)

            # 4. Back to the full sequence.
            set_subsequence(range(N_OPS))
            verify("full sequence again", acq, list(range(N_OPS)), reference)

            # 5. Repeated switching: the end PRI must not drift, the data must stay correct.
            print("[repeated switching]")
            for i in range(5):
                set_subsequence([1, 4, 9])
                verify(f"repeat {i}: [1,4,9]", acq, [1, 4, 9], reference)
                set_subsequence([2, 3, 5, 8, 13])
                verify(f"repeat {i}: [2,3,5,8,13]", acq, [2, 3, 5, 8, 13], reference)

            # 6. Timing of the set_subsequences call itself.
            same_size, diff_size = [], []
            for ops in ([1, 4, 9], [2, 5, 11], [0, 7, 15], [3, 6, 12], [5, 10, 14]):
                same_size.append(set_subsequence(ops))
                acq.collect(3)
            for ops in ([1, 4], [1, 4, 9], [1, 4, 9, 11], [1, 4], [1, 4, 9, 11, 12]):
                diff_size.append(set_subsequence(ops))
                acq.collect(3)
            print("[set_subsequences duration]")
            print(f"    same number of TX/RXs (buffer + pipeline reused): "
                  f"{', '.join(f'{v*1e3:.1f}' for v in same_size)} ms "
                  f"(median {np.median(same_size)*1e3:.1f} ms)")
            print(f"    changing number of TX/RXs (full re-creation):     "
                  f"{', '.join(f'{v*1e3:.1f}' for v in diff_size)} ms "
                  f"(median {np.median(diff_size)*1e3:.1f} ms)")

        finally:
            try:
                sess.stop_scheme()
            except Exception as e:
                print(f"stop_scheme: {e}")
            us4r.disable_hv()

    print()
    failed = [name for name, ok in results if not ok]
    print(f"PASSED: {len(results)-len(failed)}/{len(results)}")
    for name in failed:
        print(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
