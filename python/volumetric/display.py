import numpy as np
import matplotlib.pyplot as plt
import argparse
import hsdi

NSAMPLES = 2048
DTYPE = np.complex128

def main(path: str):
    print("Displaying the data")
    data, _, _ = hsdi.load_data(path)
    shape = (64, 64, 1024)
    arr = np.fromfile('pdata.bin', dtype=DTYPE)
    print(arr)
    arr = arr.reshape(shape)
    arr = arr[16:48, 16:48, :]
    arr = np.abs(arr)
    v_max = np.max(arr)
    arr = arr/v_max
    n_x, n_y, n_z = arr.shape
    dz = 0.0001230797
    pitch = 0.3e-3

    # plotting
    plane1 = arr[16, :, :].T
    plane2 = arr[:, 16, :].T

    X = np.arange(-n_x//2+1, n_x//2+1)*pitch*1000
    Y = np.arange(-n_y//2+1, n_y//2+1)*pitch*1000
    Z = np.arange(0, n_z)*dz*1000

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    print(plane1.shape)
    print(plane2.shape)
    fig.set_size_inches((3, 20))
    ax1.pcolor(
        Y, Z,
        np.clip(
            20*np.log10(plane1/np.max(plane1)),
            a_min=-40,
            a_max=0
        ),
        cmap='gray'
    )
    ax1.set_ylim([90, 0])
    ax1.set_xlim([-5, 5])
    ax2.pcolor(
        X, Z,
        np.clip(
            20*np.log10(plane2/np.max(plane2)),
            a_min=-40,
            a_max=0
        ),
        cmap='gray')
    ax2.set_ylim([90, 0])
    ax2.set_xlim([-5, 5])

    fig.savefig('test.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file", dest="file", required=True)
    args = parser.parse_args()
    main(args.file)
