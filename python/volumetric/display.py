import numpy as np
import matplotlib.pyplot as plt
import argparse
import hsdi

NSAMPLES = 2048
DTYPE = np.float64

def main(path: str):
    print("Displaying the data")
    data, _, _ = hsdi.load_data(path)
    shape = data.shape[:3] + (NSAMPLES,)
    arr = np.fromfile('pdata.bin', dtype=DTYPE)
    arr = arr.reshape(shape)
    print(np.max(np.abs(data[:, :, :, :2048]-arr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file", dest="file", required=True)
    args = parser.parse_args()
    main(args.file)
