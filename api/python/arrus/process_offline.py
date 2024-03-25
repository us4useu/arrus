import numpy as np
import pickle
import cupy as cp
import matplotlib.pyplot as plt
from arrus.utils.imaging import *

data = pickle.load(open("data.pkl", "rb"))
rf = data["rf"]
metadata = data["metadata"]

# NOTE: rf and metadata are expected to be acquired at the same stage!

x_grid = np.arange(-15, 15, 0.1) * 1e-3
z_grid = np.arange(0, 40, 0.1) * 1e-3

pipeline = Pipeline(
    steps=(
        RemapToLogicalOrder(),
        Transpose(axes=(0, 1, 3, 2)),
        BandpassFilter(),
        QuadratureDemodulation(),
        Decimation(decimation_factor=4, cic_order=2),
        ReconstructLri(x_grid=x_grid, z_grid=z_grid),
        Mean(axis=1), 
        EnvelopeDetection(),
        Mean(axis=0),
        Transpose(),
        LogCompression()
    ),
    placement="/GPU:0")

pipeline.prepare(metadata)
output = pipeline.process(cp.asarray(rf[0:1]))[0].get()
plt.imshow(output, cmap="gray", vmin=20, vmax=80)
plt.show()
