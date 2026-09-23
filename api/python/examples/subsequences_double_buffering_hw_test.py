"""Hardware test of session.prepare_subsequences (the sequencer double-buffering, MANUAL work mode).

Run with: python subsequences_double_buffering_hw_test.py --stage N
Uses the same setup and the same TX/RX "signatures" as subsequences_hw_test.py (see there): TX/RX i transmits
on the probe element ELEMENT_STEP*i only, so the RX channel profile of the TX crosstalk identifies the TX/RX
that produced a given frame.

The test is split into stages, each one enabling a bit more of the double-buffering; please run them in order:

1. MANUAL work mode only (no double-buffering): upload, run, stop, set_subsequences, run.
2. As 1, plus prepare_subsequences (the second sequencer bank and its data transfers are programmed), but the banks
   are never swapped: the scheme is stopped and the prepared sub-sequence is discarded (set_subsequences).
3. The complete flow: each iteration prepares the next sub-sequence while the scheme is running. The next run still
   acquires the data with the current sub-sequence (checked), the following ones with the prepared one.

Every step is appended to the progress log (PROGRESS_LOG, flushed and fsync-ed before the step is executed), so
that after a crash of the host it is known which step was the last one.
"""
import argparse
import os
import queue
import sys
import time
import numpy as np

import arrus
import arrus.session
import arrus.logging
from arrus.ops.us4r import Scheme, DataBufferSpec
from arrus.utils.imaging import Pipeline, Processing, RemapToLogicalOrder

from subsequences_hw_test import (CFG, VOLTAGE, N_OPS, get_sequence, rx_profiles, identify_ops, check, results)

arrus.set_clog_level(arrus.logging.ERROR)

N_SELECTED = 3
RX_BUFFER_SIZE = 2
N_ITERATIONS = 40
FRAMES_PER_ITERATION = 8
TIMEOUT = 5.0
PROGRESS_LOG = os.environ.get("PROGRESS_LOG", os.path.expanduser("~/subsequences_double_buffering_progress.log"))


class Progress:
    """Appends each step to the progress log, and makes sure it is on the disk before the step is executed."""

    def __init__(self, path):
        self.file = open(path, "a")

    def __call__(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    args = parser.parse_args()
    step = Progress(PROGRESS_LOG)
    step(f"=== stage {args.stage}, cfg: {CFG}, voltage: {VOLTAGE} V, arrus: {arrus.__file__}")

    frames = queue.Queue()

    def on_frame(element):
        frames.put(np.asarray(element.arrays[0], dtype=np.float32).copy())
        element.release()

    pipeline = Pipeline(steps=(RemapToLogicalOrder(), ), placement="/GPU:0")
    processing = Processing(graph=pipeline, callback=on_frame)

    def acquire(sess, n=1, name=""):
        """Triggers n frames (one by one), returns the average of |data| (a single crosstalk is noisy)."""
        result = []
        for i in range(n):
            step(f"{name}: run {i}")
            sess.run(sync=True, timeout=int(TIMEOUT*1000))
            frame = frames.get(timeout=TIMEOUT)
            result.append(np.abs(frame[0] if frame.ndim == 4 else frame))
        return np.mean(np.stack(result), axis=0)

    def drain():
        while not frames.empty():
            frames.get_nowait()

    rng = np.random.default_rng(0)

    def random_ops():
        return sorted(rng.choice(N_OPS, N_SELECTED, replace=False).tolist())

    step("opening the session")
    with arrus.Session(CFG) as sess:
        us4r = sess.get_device("/Us4R:0")
        n_elements = us4r.get_probe_model().n_elements
        step(f"session opened, probe: {n_elements} elements; setting HV")
        us4r.set_hv_voltage(VOLTAGE)
        try:
            scheme = Scheme(
                tx_rx_sequence=get_sequence(n_elements),
                rx_buffer_size=RX_BUFFER_SIZE,
                # prepare_subsequences requires the output buffer size equal to the RX buffer size.
                output_buffer=DataBufferSpec(type="FIFO", n_elements=RX_BUFFER_SIZE),
                work_mode="MANUAL",
                processing=processing,
            )
            step("upload")
            sess.upload(scheme)
            us4r.set_stop_on_overflow(False)
            # Reference RX profiles: the full sequence.
            reference = rx_profiles(acquire(sess, 8, "reference"))
            step("stop")
            sess.stop_scheme()
            drain()

            ops = random_ops()
            step(f"set_subsequences({ops}) (stopped)")
            sess.set_subsequences([ops], processing=processing)
            prepare_times, run_times = [], []
            for i in range(args.iterations):
                t = time.time()
                # A few frames of the same sub-sequence: a single acquisition of the crosstalk is noisy.
                frame = acquire(sess, FRAMES_PER_ITERATION, f"iteration {i}, ops {ops}")
                run_times.append((time.time()-t)/FRAMES_PER_ITERATION)
                detected = identify_ops(frame, reference)
                check(f"iteration {i}: ops {ops}", np.array_equal(detected, np.asarray(ops)),
                      f"identified {list(detected)}")
                next_ops = random_ops()
                if args.stage == 1:
                    step("stop")
                    sess.stop_scheme()
                    drain()
                    step(f"set_subsequences({next_ops}) (stopped)")
                    sess.set_subsequences([next_ops], processing=processing)
                elif args.stage == 2:
                    step(f"prepare_subsequences({next_ops}) (running, will be discarded)")
                    t = time.time()
                    sess.prepare_subsequences([next_ops], processing=processing)
                    prepare_times.append(time.time()-t)
                    step("stop")
                    sess.stop_scheme()
                    drain()
                    step(f"set_subsequences({next_ops}) (stopped, discards the prepared one)")
                    sess.set_subsequences([next_ops], processing=processing)
                else:
                    step(f"prepare_subsequences({next_ops}) (running)")
                    t = time.time()
                    sess.prepare_subsequences([next_ops], processing=processing)
                    prepare_times.append(time.time()-t)
                    # The next run still acquires the data with the current sub-sequence.
                    transition = acquire(sess, 1, f"iteration {i}, transition frame (expected: {ops})")
                    profiles = rx_profiles(transition)
                    score_current = np.mean(np.sum(profiles*reference[ops], axis=1))
                    score_next = np.mean(np.sum(profiles*reference[next_ops], axis=1))
                    check(f"iteration {i}: the transition frame is acquired with the current sub-sequence",
                          score_current > score_next,
                          f"similarity to {ops}: {score_current:.3f}, to {next_ops}: {score_next:.3f}")
                ops = next_ops
            if prepare_times:
                step(f"prepare_subsequences: median {np.median(prepare_times)*1e3:.2f} ms, "
                     f"max {np.max(prepare_times)*1e3:.2f} ms")
            step(f"run + frame: median {np.median(run_times)*1e3:.2f} ms, max {np.max(run_times)*1e3:.2f} ms")

            if args.stage == 3:
                # A sub-sequence with a different number of TX/RXs cannot be prepared while running.
                step("prepare_subsequences with a different number of TX/RXs (expected to be rejected)")
                try:
                    sess.prepare_subsequences([list(range(N_SELECTED+1))], processing=processing)
                    check("different number of TX/RXs is rejected", False)
                except Exception as e:
                    check("different number of TX/RXs is rejected", True, str(e).splitlines()[0])
        finally:
            step("stop + HV off")
            try:
                sess.stop_scheme()
            except Exception as e:
                print(f"stop_scheme: {e}")
            us4r.disable_hv()
    step("session closed")

    failed = [name for name, ok in results if not ok]
    step(f"PASSED: {len(results)-len(failed)}/{len(results)}")
    for name in failed:
        step(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
