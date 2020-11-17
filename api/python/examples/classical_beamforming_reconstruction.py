import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

import arrus
import arrus.ops.us4r
from arrus.utils.us4r import RemapToLogicalOrder
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


def main():
    parser = argparse.ArgumentParser(
        description="The script reconstructs image from the given data.")

    parser.add_argument("--rf", dest="rf", help="Path to rf data file.",
                        required=False)
    parser.add_argument("--metadata", dest="metadata",
                        help="Path to metadata file.",
                        required=False)
    parser.add_argument("--matlab_dataset", dest="matlab_dataset",
                        help="Path to matlab dataset (v0.4.7).",
                        required=False)
    args = parser.parse_args()

    is_numpy_data = args.rf is not None and args.metadata is not None
    is_matlab_data = args.matlab_dataset is not None
    if not (is_numpy_data ^ is_matlab_data):
        raise ValueError("Exactly one of the following datasets should be "
                         "provided: numpy rf data and metadata, "
                         "matlab dataset.")

    if is_numpy_data:
        data = np.load(args.rf)
        metadata = pickle.load(open(args.metadata, 'rb'))
        mock = {}  # No matlab mock
    elif is_matlab_data:
        import h5py
        dataset = h5py.File("data.mat", mode="r")
        dataset = {
            "rf": np.array(dataset["rf"][:5, :, :, :]),
            "sys": dataset["sys"],
            "seq": dataset["seq"]
        }
        mock = {"Us4R:0": dataset}

    # Create session with not Us4R device.
    sess = arrus.Session(mock=mock)
    gpu = sess.get_device("/GPU:0")

    if is_numpy_data:
        initial_steps = [
            RemapToLogicalOrder(),
            Transpose(axes=(0, 2, 1))
        ]
    elif is_matlab_data:
        seq = arrus.ops.us4r.TxRxSequence([], [])
        us4r = sess.get_device("/Us4R:0")
        buffer = us4r.upload(seq)
        data, metadata = buffer.tail()
        initial_steps = []

    # Actual imaging starts here.
    x_grid = np.arange(-50, 50, 0.4)*1e-3
    z_grid = np.arange(0, 60, 0.4)*1e-3

    pipeline = Pipeline(
        steps=initial_steps + [
            BandpassFilter(),
            QuadratureDemodulation(),
            Decimation(decimation_factor=4, cic_order=2),
            RxBeamforming(),
            EnvelopeDetection(),
            Transpose(),
            ScanConversion(x_grid=x_grid, z_grid=z_grid),
            LogCompression(),
            DynamicRangeAdjustment(),
            ToGrayscaleImg()
        ],
        placement=gpu)
    reconstructed_data, metadata = pipeline(cp.asarray(data), metadata)

    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.imshow(reconstructed_data, cmap="gray")
    ax.set_aspect('auto')
    fig.show()
    plt.show()


if __name__ == "__main__":
    main()



