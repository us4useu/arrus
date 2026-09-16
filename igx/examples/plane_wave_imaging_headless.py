"""
api/python/examples/plane_wave_imaging.py without the window: acquires 12 B-mode frames, prints
the steady frame rate and saves the last frame as plane_wave_imaging.png. The headless
acceptance step for a machine without a display; the sequence, scheme and pipeline are the
example's own. GPU is required. Needs us4r.prototxt in the working directory.
"""
import numpy as np
import arrus
from arrus.ops.us4r import Scheme, Pulse, DataBufferSpec
from arrus.ops.imaging import PwiSequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from arrus.utils.imaging import get_bmode_imaging, get_extent

arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

# Here starts communication with the device.
with arrus.Session("us4r.prototxt") as sess:
    us4r = sess.get_device("/Us4R:0")
    us4r.set_hv_voltage(20)

    sequence = PwiSequence(
        angles=np.linspace(-10, 10, 32)*np.pi/180,
        pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
        rx_sample_range=(0, 1024*4),
        downsampling_factor=1,
        speed_of_sound=1450,
        pri=200e-6,
        tgc_start=14,
        tgc_slope=2e2)

    # Imaging output grid.
    x_grid = np.arange(-15, 15, 0.1) * 1e-3
    z_grid = np.arange(5, 35, 0.1) * 1e-3

    scheme = Scheme(
        tx_rx_sequence=sequence,
        processing=get_bmode_imaging(sequence=sequence, grid=(x_grid, z_grid)))
    # Upload sequence on the us4r-lite device.
    buffer, metadata = sess.upload(scheme)
    sess.start_scheme()
    t0 = time.time(); n = 0
    for i in range(12):
        datas = buffer.get(timeout=240 if i == 0 else 60)
        img = np.asarray(datas[0]); n += 1
        if i == 1: t1 = time.time()
    dt = time.time() - t1
    sess.stop_scheme()
    print(f"got {n} B-mode frames; steady rate {(n - 2) / dt:.1f} fps over the last {n - 2}; image shape {img.shape}, range {img.min():.1f}..{img.max():.1f} dB")
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(img, cmap="gray", vmin=20, vmax=80, extent=get_extent(x_grid, z_grid) * 1e3)
    ax.set_title(f"plane_wave_imaging.py over Ethernet (RDMA): B-mode, SL1543, 32 angles, HV 20 V"); ax.set_xlabel("OX (mm)"); ax.set_ylabel("OZ (mm)")
    fig.colorbar(im, ax=ax, fraction=0.04); fig.tight_layout(); fig.savefig("plane_wave_imaging.png", dpi=110)
    print("saved plane_wave_imaging.png")

# When we exit the above scope, the session and scheme is properly closed.
print("Stopping the example.")
