import numpy as np
import hsdi
import argparse
import matplotlib.pyplot as plt

NSAMPLES = 2048

def save_binary_data(path: str):
    print("Reading input data")
    data, acq_params, sys_params = hsdi.load_data(path)
    data = data[:, :, :, :2048]

    plt.imshow(data[0, 32//2, :, :256], cmap='gray')
    plt.savefig('test_input.png')

    print("Data shape: %s" % str(data.shape))
    print("Data type: %s" % str(data.dtype))
    data.tofile("data.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file", dest="file", required=True)
    args = parser.parse_args()
    save_binary_data(args.file)

