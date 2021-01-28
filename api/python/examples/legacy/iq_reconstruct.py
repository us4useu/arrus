import arrus
import numpy as np
import arrus.utils.us4r
import pickle
import cupy as cp

from arrus.utils.imaging import (
    Pipeline,
    BandpassFilter,
    QuadratureDemodulation,
    Decimation,
    RxBeamforming,
    EnvelopeDetection,
    Transpose,
    ScanConversion,
    LogCompression,
    DynamicRangeAdjustment,
    ToGrayscaleImg
)

from arrus.utils.us4r import RemapToLogicalOrder, get_batch_data, get_batch_metadata

iq_reconstruct = Pipeline(
    steps=(
        RemapToLogicalOrder(),
        Transpose(axes=(0, 2, 1)),
        BandpassFilter(),
        QuadratureDemodulation(),
        Decimation(decimation_factor=4, cic_order=2),
        RxBeamforming()))


data = np.load(f"C:\\Users\\pjarosik\\Desktop\\test_rf.npy")
metadata = pickle.load(open(f"C:\\Users\\pjarosik\\Desktop\\test_metadata.pkl", 'rb'))

session = arrus.session.Session(mock={})
gpu = session.get_device("/GPU:0")


# Set the pipeline to be executed on the GPU
iq_reconstruct.set_placement(gpu)

raw_data = []

for i in range(1000):
    j = i % 100
    iq_data, iq_metadata = iq_reconstruct(cp.asarray(data[j]), metadata[j])

