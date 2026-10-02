"""
AFE58JD32 (us4OEM+ 64): odd/even input sampling skew, before and after correction.

Each AFE58JD32 converter samples its odd input (raw row 0) and its even input
(raw row 1) on alternate 65 MHz ADC clocks (SBAS823A 8.3.10.1): the raw row 1
channels are sampled 1/65 MHz = 15.4 ns (half a sample at 32.5 MHz) later.
The correction delays the raw row 1 channels by half a sample (windowed-sinc
fractional delay FIR).

Setup: a plane wave (angle 0, full TX aperture) reflected from a flat reflector.

Usage:
    python jd32_skew_correction.py              # acquire, save, analyze
    python jd32_skew_correction.py --offline    # analyze the saved file only

Shows:
- two neighbouring RF channels (one raw row 0, one raw row 1) around the
  reflector echo, before and after the correction; the fitted reflector shape
  (geometric delay between the two elements) is removed from the second one,
  so only the skew is left: ~half a sample apart before, on top of each other after,
- the echo lag between neighbouring elements, coloured by the raw row change, with the
  reflector shape removed (a smooth polynomial across the aperture, fitted together with
  the skew; pairs off by > 3 sigma, e.g. cycle skips, are dropped and drawn as "+"); the estimated
  row 1 - row 0 skew should be about -15.4 ns before (row 1 sees the echo earlier in sample index)
  and ~0 after.

Interactive: click a pair in the bottom plot (or use the left / right keys) to show it in the RF plots.
The RF plots mark the echo peak (v) and the rising zero crossing (diamond) of each curve and print their
time differences; a left / right click in an RF plot sets measurement cursor A / B ('c' clears them).
"""
import argparse
import os
import threading
import time

import numpy as np

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jd32_skew_capture.npz")
N_SAMPLES = 2048          # 63 us at 32.5 MHz, i.e. ~47 mm at 1490 m/s
PRI = 200e-6
HV_VOLTAGE = 5            # [V] (minimum)
LNA_GAIN = 12             # [dB] AFE58JD32 total gain (min. 12, max. 51; PGA gain must stay 0)
N_ELEMENTS_TO_SAVE = 4    # Buffer elements (sequences) to save; the analysis averages them.
SKIP_SAMPLES = 150        # TX bang / near field, ignored when looking for the reflector echo.
N_TAPS = 32               # Fractional delay FIR length.
ZOOM_HALF_WIDTH = 8       # RF plots: +/- samples around the echo peak.
UPSAMPLE = 64             # RF plots: interpolation factor of the drawn curves and markers (0.5 ns).
FIT_DEGREE = 8            # Reflector shape (lag vs element) polynomial degree in the skew fit.
ADC_CLOCK = 65e6


def acquire():
    import arrus
    import arrus.medium
    import arrus.session
    from arrus.ops.us4r import Scheme, Pulse, Tx, Rx, TxRx, TxRxSequence, DataBufferSpec

    arrus.set_clog_level(arrus.logging.INFO)
    arrus.add_log_file("jd32_skew_correction.log", arrus.logging.INFO)

    frames, lock, done = [], threading.Lock(), threading.Event()

    def callback(element):
        with lock:
            if len(frames) < N_ELEMENTS_TO_SAVE:
                frames.append(np.array(element.data, copy=True))
                if len(frames) == N_ELEMENTS_TO_SAVE:
                    done.set()
        element.release()

    medium = arrus.medium.Medium(name="water", speed_of_sound=1490)
    with arrus.Session("us4oem.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(HV_VOLTAGE)
        us4r.set_lna_gain(LNA_GAIN)
        n_elements = us4r.get_probe_model().n_elements
        seq = TxRxSequence(
            ops=[TxRx(
                Tx(aperture=[True] * n_elements,
                   excitation=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
                   focus=np.inf, speed_of_sound=1490, angle=0),
                Rx(aperture=[True] * n_elements, sample_range=(0, N_SAMPLES), downsampling_factor=1),
                pri=PRI)],
            tgc_curve=[], sri=50e-3)
        scheme = Scheme(tx_rx_sequence=seq, rx_buffer_size=4,
                        output_buffer=DataBufferSpec(type="FIFO", n_elements=4), work_mode="HOST")
        buffer, metadata = sess.upload(scheme)
        buffer.append_on_new_data_callback(callback)
        sess.start_scheme()
        time.sleep(0.5)
        if not done.wait(10.0):
            raise TimeoutError(f"Got only {len(frames)} of {N_ELEMENTS_TO_SAVE} buffer elements.")
        sess.stop_scheme()
        md = metadata[0] if isinstance(metadata, (list, tuple)) else metadata
        fcm = md.data_description.custom["frame_channel_mapping"]
        results = dict(rf=np.stack(frames),
                       fcm_frames=np.array(fcm.frames), fcm_channels=np.array(fcm.channels),
                       sampling_frequency=us4r.sampling_frequency, n_samples=N_SAMPLES, pri=PRI)
    # Write after the session is closed (a long write inside the session can starve the host watchdog).
    np.savez_compressed(OUTPUT_FILE, **results)
    print(f"Saved: {OUTPUT_FILE}")


def to_element_order(d):
    """Raw buffer -> (n_elements, n_samples) for the (single) TX/RX op, plus the raw row of each element."""
    rf, ns = d["rf"].astype(np.float64).mean(axis=0), int(d["n_samples"])   # average the saved sequences
    ff, fc = d["fcm_frames"][0], d["fcm_channels"][0]
    n_el = ff.shape[0]
    data = np.stack([rf[int(ff[e]) * ns:(int(ff[e]) + 1) * ns, int(fc[e])] for e in range(n_el)])
    data -= data[:, SKIP_SAMPLES:].mean(axis=1, keepdims=True)              # remove DC
    data[:, 0] = 0                                                          # frame header row
    rows = np.array([int(fc[e]) // 32 for e in range(n_el)])
    groups = np.array([int(ff[e]) for e in range(n_el)])                    # firing (RX mux group)
    return data, rows, groups


def half_sample_delay_fir(n_taps=N_TAPS):
    """Windowed-sinc FIR with group delay (n_taps/2 - 1) + 0.5 samples."""
    k = np.arange(n_taps) - (n_taps / 2 - 1) - 0.5
    h = np.sinc(k) * np.kaiser(n_taps, 6.0)
    return h / h.sum()


def correct(data, rows):
    """Delay raw row 1 channels by 0.5 sample (they are sampled 15.4 ns late, so an echo appears 0.5 sample
    early in their sample index); both rows get the same integer delay of the FIR."""
    h = half_sample_delay_fir()
    d_int = N_TAPS // 2 - 1
    out = np.empty_like(data)
    for e in range(data.shape[0]):
        if rows[e] == 1:
            out[e] = np.convolve(data[e], h)[:data.shape[1]]
        else:
            out[e] = np.concatenate([np.zeros(d_int), data[e][:-d_int]])
    return out


def envelope(x):
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    hgain = np.zeros(n)
    hgain[0] = 1
    hgain[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        hgain[n // 2] = 1
    return np.abs(np.fft.ifft(X * hgain, axis=-1))


def lag(x, ref, upsample=32, max_lag=3.0):
    """Sub-sample delay of x relative to ref [samples] (positive: x arrives later), FFT cross-correlation,
    searched within +/- max_lag samples (neighbouring elements: no cycle skips)."""
    n = len(x)
    X, R = np.fft.rfft(x, 2 * n), np.fft.rfft(ref, 2 * n)
    cc = np.fft.fftshift(np.fft.irfft(X * np.conj(R), 2 * n * upsample))
    lags = (np.arange(len(cc)) - n * upsample) / upsample
    m = np.abs(lags) <= max_lag
    return lags[m][np.argmax(cc[m])]


def neighbour_lags(x, rows, idx, window):
    """For neighbouring elements (e, e+1) of one firing: lag of e+1 vs e and the raw row change."""
    lags = np.array([lag(x[e + 1, window], x[e, window]) for e in idx[:-1]])
    drow = np.array([rows[e + 1] - rows[e] for e in idx[:-1]], dtype=float)
    return lags, drow


def fit_skew(lags, drow, degree=FIT_DEGREE, n_iter=3, n_sigma=3.0):
    """lag(e+1 vs e) = shape(e) + skew * (row[e+1] - row[e]), shape(e): polynomial in the element position
    (reflector tilt / curvature, wavefront shape). Pairs off by more than n_sigma (cycle skips) are dropped and
    the fit is repeated. -> skew [samples], residual = lag - shape (shape removed), outlier mask."""
    x = np.linspace(-1, 1, len(lags))
    A = np.column_stack([x ** k for k in range(degree + 1)] + [drow])
    ok = np.ones(len(lags), dtype=bool)
    for _ in range(n_iter):
        coeffs, *_ = np.linalg.lstsq(A[ok], lags[ok], rcond=None)
        r = lags - A @ coeffs
        ok = np.abs(r) <= n_sigma * np.std(r[ok]) + 1e-9
    coeffs, *_ = np.linalg.lstsq(A[ok], lags[ok], rcond=None)  # final fit with the returned mask
    shape = A[:, :-1] @ coeffs[:-1]
    return coeffs[-1], lags - shape, ok


def fractional_delay(x, delay):
    """Delays x by delay samples (may be fractional / negative), FFT phase shift."""
    n = len(x)
    f = np.fft.rfftfreq(2 * n)
    return np.fft.irfft(np.fft.rfft(x, 2 * n) * np.exp(-2j * np.pi * f * delay), 2 * n)[:n]


def upsample(x, factor):
    """Band-limited (FFT zero padding) interpolation -> (sample positions, values)."""
    n = len(x)
    X = np.fft.rfft(x)
    y = np.fft.irfft(X, n * factor) * factor
    return np.arange(n * factor) / factor, y


def echo_window(data, idx, shift=0, half_width=40):
    env = envelope(data[idx]).mean(axis=0)
    peak = SKIP_SAMPLES + int(np.argmax(env[SKIP_SAMPLES:]))
    return slice(max(peak - half_width, SKIP_SAMPLES) + shift, min(peak + half_width, data.shape[1]) + shift)


def analyze():
    import matplotlib.pyplot as plt

    d = np.load(OUTPUT_FILE)
    fs = float(d["sampling_frequency"])
    data, rows, groups = to_element_order(d)
    corrected = correct(data, rows)
    n_el, ns = data.shape
    d_int = N_TAPS // 2 - 1

    # Skew per RX mux group (each group is one firing), from neighbouring element pairs; the reflector shape
    # (tilt, curvature) varies smoothly across the aperture and is removed by the polynomial term.
    results = {}
    for name, x, shift in (("before", data, 0), ("after", corrected, d_int)):
        skews, pairs = [], []
        for g in np.unique(groups):
            idx = np.nonzero(groups == g)[0]
            w = echo_window(data, idx, shift)
            lags, drow = neighbour_lags(x, rows, idx, w)
            skew, residual, ok = fit_skew(lags, drow)
            skews.append(skew)
            pairs.append((idx[:-1], drow, residual, ok, lags - residual))
        results[name] = (np.mean(skews), pairs)
        n_out = sum(int(np.sum(~p[3])) for p in pairs)
        # Scatter of what the fit does not explain (shape and skew removed).
        spread = np.std(np.concatenate([(p[2] - s * p[1])[p[3]] for p, s in zip(pairs, skews)])) / fs * 1e9
        print(f"{name:6s}: row 1 - row 0 arrival = {np.mean(skews) / fs * 1e9:+6.2f} ns "
              f"(per mux group: {', '.join(f'{s / fs * 1e9:+.2f}' for s in skews)} ns); "
              f"pair scatter {spread:.2f} ns, {n_out} outlier pairs dropped; "
              f"physics: row 1 sampled {1e9 / ADC_CLOCK:.2f} ns later -> {-1e9 / ADC_CLOCK:+.2f} ns before correction")

    # Reflector shape per pair (tilt/curvature, no skew; the "after" fit, the shape is the same in both fits): it
    # is removed from e+1 in the RF plots, so that only the skew remains visible.
    pair_info = {}  # e -> (group echo window, shape lag [samples])
    for (elems, *_, shape), g in zip(results["after"][1], np.unique(groups)):
        w = echo_window(data, np.nonzero(groups == g)[0])
        for e, s in zip(elems, shape):
            pair_info[int(e)] = (w, s)
    pair_list = sorted(pair_info)

    # Default: neighbouring channels of different raw rows near the aperture centre of mux group 0.
    group0 = np.nonzero(groups == groups[0])[0]
    centre = group0[len(group0) // 2]
    first = next(e for e in range(centre, group0[-1]) if rows[e] != rows[e + 1])

    # Interactive view: a click in the bottom plot / the left-right keys select the pair; a left / right click in
    # an RF plot sets measurement cursor A / B (shown in both RF plots). Automatic markers: the echo peak and the
    # rising zero crossing next to it, for each curve, and their time difference (e+1 minus e).
    for km in ("keymap.back", "keymap.forward"):  # free the keys used below (default: toolbar back / forward)
        plt.rcParams[km] = [k for k in plt.rcParams[km] if k not in ("left", "right", "c")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    rf_axes = axes[:2]
    state = {"e0": first, "cursors": {}}

    def t_ns(samples):
        return samples / fs * 1e9

    def markers(t_fine, x_fine, centre_t, ref_zc=None):
        """-> (peak time, peak value, rising zero crossing time) [samples].
        ref_zc None: the highest peak within half a period of centre_t and the rising crossing nearest to it.
        Otherwise (same cycle as a reference): the local maximum nearest to centre_t and the rising crossing
        nearest to ref_zc."""
        if ref_zc is None:
            near = np.abs(t_fine - centre_t) <= fs / 6e6 / 2
            i_pk = np.nonzero(near)[0][np.argmax(x_fine[near])]
        else:
            loc = np.nonzero((x_fine[1:-1] > x_fine[:-2]) & (x_fine[1:-1] >= x_fine[2:]))[0] + 1
            i_pk = loc[np.argmin(np.abs(t_fine[loc] - centre_t))]
        zc = np.nonzero((x_fine[:-1] < 0) & (x_fine[1:] >= 0))[0]
        zc_t = t_fine[zc] - x_fine[zc] * (t_fine[zc + 1] - t_fine[zc]) / (x_fine[zc + 1] - x_fine[zc])
        target = t_fine[i_pk] if ref_zc is None else ref_zc
        t_zc = zc_t[np.argmin(np.abs(zc_t - target))] if len(zc_t) and not np.isnan(target) else np.nan
        return t_fine[i_pk], x_fine[i_pk], t_zc

    def draw_cursors():
        for ax in rf_axes:
            for artist in [a for a in list(ax.lines) + list(ax.texts) if a.get_gid() == "cursor"]:
                artist.remove()
        cur = state["cursors"]
        for label, t in cur.items():
            color = "k" if label == "A" else "tab:red"
            for ax in rf_axes:
                ax.axvline(t, color=color, lw=1, gid="cursor")
                ax.text(t, 1.0, label, color=color, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=9, gid="cursor")
        if len(cur) == 2:
            rf_axes[0].text(0.01, 0.88, f"cursors: B - A = {(cur['B'] - cur['A']) * 1e3:+.1f} ns",
                            transform=rf_axes[0].transAxes, fontsize=10, bbox=dict(fc="lightyellow", ec="0.7"),
                            gid="cursor")

    def draw_pair():
        e0 = state["e0"]
        e1 = e0 + 1
        window, shape_lag = pair_info[e0]
        env = envelope(data[[e0, e1]]).sum(axis=0)
        peak = window.start + int(np.argmax(env[window]))
        zoom = slice(peak - ZOOM_HALF_WIDTH, peak + ZOOM_HALF_WIDTH + 1)
        for ax, x, shift, name in ((rf_axes[0], data, 0, "before"), (rf_axes[1], corrected, d_int, "after")):
            ax.clear()
            x0 = x[e0][shift:]
            x1 = fractional_delay(x[e1], -shape_lag)[shift:]
            pair_lag = lag(x1[window], x0[window])
            curves = [(t, zoom.start + t_s) for t_s, t in (upsample(x0[zoom], UPSAMPLE), upsample(x1[zoom], UPSAMPLE))]
            # The same cycle for both curves and both panels: chosen on the sum of the two curves before the
            # correction, then each curve's own peak / crossing nearest to it.
            if name == "before":
                ref_pk, _, ref_zc = markers(curves[0][1], curves[0][0] + curves[1][0], peak)
            found = []
            for (x_fine, t_fine), xe, e, marker, color, extra in (
                    (curves[0], x0, e0, "o", "tab:blue", ""),
                    (curves[1], x1, e1, "s", "tab:orange", f", reflector shape {t_ns(shape_lag):+.1f} ns removed")):
                ax.plot(t_fine / fs * 1e6, x_fine, "-", color=color, lw=1.2)
                ax.plot(np.arange(zoom.start, zoom.stop) / fs * 1e6, xe[zoom], marker, color=color, ms=4,
                        label=f"element {e} (raw row {rows[e]}){extra}")
                t_pk, v_pk, t_zc = markers(t_fine, x_fine, ref_pk, ref_zc)
                ax.plot(t_pk / fs * 1e6, v_pk, "v", color=color, ms=10, mec="k")
                ax.plot(t_zc / fs * 1e6, 0, "D", color=color, ms=7, mec="k")
                ax.axvline(t_zc / fs * 1e6, color=color, ls=":", lw=1)
                found.append((t_pk, t_zc))
            d_pk, d_zc = t_ns(found[1][0] - found[0][0]), t_ns(found[1][1] - found[0][1])
            ax.text(0.01, 0.04, f"{e1} - {e0}:  peak (v) {d_pk:+.1f} ns,  zero crossing (diamond) {d_zc:+.1f} ns,  "
                                f"cross-correlation {t_ns(pair_lag):+.1f} ns",
                    transform=ax.transAxes, fontsize=9, bbox=dict(fc="w", ec="0.7"))
            ax.set_title(f"{name} correction: elements {e0} / {e1} "
                         f"(all elements: row 1 - row 0 skew {t_ns(results[name][0]):+.2f} ns)")
            ax.set_xlabel("time [us]" + (f" (FIR delay of {d_int} samples removed)" if name == "after" else ""))
            ax.set_ylabel("amplitude")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.6)  # peak markers are at the top
            ax.grid(True)
        draw_cursors()
        selection.set_xdata([e0 + 0.5, e0 + 0.5])
        fig.canvas.draw_idle()

    def on_click(event):
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return  # zoom / pan active
        if event.xdata is None:
            return
        if event.inaxes is axes[2]:
            state["e0"] = min(pair_list, key=lambda p: abs(p + 0.5 - event.xdata))
            draw_pair()
        elif event.inaxes in rf_axes and event.button in (1, 3):
            state["cursors"]["A" if event.button == 1 else "B"] = event.xdata
            draw_cursors()
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("left", "right"):
            i = pair_list.index(state["e0"]) + (1 if event.key == "right" else -1)
            state["e0"] = pair_list[int(np.clip(i, 0, len(pair_list) - 1))]
            draw_pair()
        elif event.key == "c":
            state["cursors"].clear()
            draw_cursors()
            fig.canvas.draw_idle()

    colors = {1.0: "tab:orange", -1.0: "tab:blue", 0.0: "tab:gray"}
    labels = {1.0: "row 0 -> 1", -1.0: "row 1 -> 0", 0.0: "same row"}
    for name, marker, alpha in (("before", "o", 0.35), ("after", "x", 1.0)):
        for elems, drow, residual, ok, _ in results[name][1]:
            for dr in (1.0, -1.0, 0.0):
                m = (drow == dr) & ok
                axes[2].plot(elems[m] + 0.5, residual[m] / fs * 1e9, marker, color=colors[dr], ms=4, alpha=alpha,
                             label=f"{name}, {labels[dr]}")
            axes[2].plot(elems[~ok] + 0.5, residual[~ok] / fs * 1e9, "+", color="tab:red", ms=8, alpha=alpha,
                         label="outlier (dropped)")
    handles, labs = axes[2].get_legend_handles_labels()
    uniq = dict(zip(labs, handles))
    axes[2].legend(uniq.values(), uniq.keys(), ncol=2, fontsize=8)
    selection = axes[2].axvline(first + 0.5, color="k", lw=6, alpha=0.15)
    axes[2].set_title("echo lag between neighbouring elements (e+1 vs e), reflector shape removed\n"
                      "click: plot this pair above;  left / right key: previous / next pair;  "
                      "RF plots: left / right click = cursor A / B, 'c' clears", fontsize=10)
    axes[2].set_xlabel("element pair (e, e+1)")
    axes[2].set_ylabel("lag [ns]")
    axes[2].grid(True)
    draw_pair()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="analyze the saved file only")
    args = parser.parse_args()
    if not args.offline:
        acquire()
    analyze()
