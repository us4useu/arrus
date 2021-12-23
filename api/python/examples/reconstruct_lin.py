import arrus
import pickle
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from arrus.utils.imaging import (
    get_bmode_imaging, get_extent,
    RxBeamforming,
    ScanConversion,
    Pipeline,
    RemapToLogicalOrder,
    SelectSequence,
    Squeeze,
    Lambda,
    Transpose,
    BandpassFilter,
    QuadratureDemodulation,
    Decimation,
    ReconstructLri,
    Mean,
    EnvelopeDetection,
    LogCompression
)

data = pickle.load(open("classical_data.pkl", "rb"))
bmode, rf = data["data"]
bmode_metadata, rf_metadata = data["metadata"]

x_grid = np.arange(-15, 15, 0.1) * 1e-3
z_grid = np.arange(5, 35, 0.1) * 1e-3

pipeline = Pipeline(
    steps=(
        # Channel data pre-processing.
        RemapToLogicalOrder(),
        Transpose(axes=(0, 1, 3, 2)),
        BandpassFilter(),
        QuadratureDemodulation(),
        Lambda(lambda data: (print(data.shape), data)[1]),
        # Decimation(decimation_factor=4, cic_order=2),
        # # Data beamforming.
        # RxBeamforming(),
        # # Post-processing to B-mode image.
        # EnvelopeDetection(),
        # Transpose(axes=(0, 2, 1)),
        # ScanConversion(x_grid, z_grid),
        # LogCompression()
    ),
    placement="/GPU:0")

print(rf.shape)
pipeline.prepare(rf_metadata)
bmode = pipeline.process(cp.asarray(rf))[0].get()
print(bmode.shape)

plt.imshow(bmode, cmap="gray")
plt.show()
