from dataclasses import dataclass
import re
import pathlib
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load mat file and data.
@dataclass(frozen=True)
class Probe:
    pitch: np.float64
    kerf: np.float32

@dataclass(frozen=True)
class SystemParameters:
    probe: Probe

@dataclass(frozen=True)
class AcquisitionParameters:
    tx_frequency: np.float64
    sampling_frequency: np.float64
    speed_of_sound: np.float64
    rx_aperture: (np.uint16, np.uint16)

def load_data(path: str):
    """
    :param path: path to a file to load
    :return: (rf, acquisition_parameters, system_parameters)
        RF data is in format ECCS (emission, channel, channel, sample)
    """
    rf = None
    with h5py.File(path, "r") as f:
        rf = f["Matrix"][:]
    filename = pathlib.Path(path).name
    dims = re.findall(r"TXRX(\d+)x(\d+)\.mat", filename)
    if len(dims) > 0 and len(dims[0]) != 2:
        raise ValueError(r"Input file should be of format: \w+TXRX\d+x\d+")
    rx_aperture_size_x, rx_aperture_size_y = dims[0]
    acq_parameters = AcquisitionParameters(
        # based on the provided script
        tx_frequency=3.3e6,
        sampling_frequency=25e6,
        speed_of_sound=1540,
        rx_aperture=(rx_aperture_size_x, rx_aperture_size_y)
    )
    pitch = 0.3e-3
    sys_parameters = SystemParameters(
        probe=Probe(
            pitch=pitch,
            kerf=0.2*pitch
        )
    )
    return rf, acq_parameters, sys_parameters


def perform_tgc():
    raise NotImplementedError


def reconstruct_hri(rf, acq_params, sys_params, n_z):
    """
    Reconstructs volume data ("High Resolution Image")
    from given RF frame using HSDI approach.

    :param rf: RF data
    :param acq_params: parameters of the acquisiton procedure
    :param sys_params: parameters of the system, which acquired the data
    :param n_z: number of points along OZ axis
    :return: a volume of shape (nx, ny, nz)
    """
    n_emissions, n_x, n_y, n_samples = rf.shape
    hri = np.zeros(shape=(n_x, n_y, n_z), dtype=np.complex64)
    for emission in range(n_emissions):
        print("Emission: %d" % emission, end='\r')
        hri += reconstruct_lri(
            rf[emission, :, :, :],
            acq_params,
            sys_params,
            n_z
        )
    return hri


def reconstruct_lri(rf, acq_params, sys_params, n_z):
    """
    Reconstructs single low-resolution image from the provided RF signal data
    acquired from a single emission.
    """
    n_x, n_y, n_samples = rf.shape

    # (OX, OY) zero padding parameters
    padding = 2
    padded_n_x, padded_n_y = padding*n_x, padding*n_y
    if n_x % 2 != 0 or n_y % 2 != 0 or padding % 2 != 0:
        raise ValueError("Each of: n_x=%d, n_y=%d, padding=%d, should be "
                         "divisible by 2"
                         %(n_x, n_y, padding))
    padded_center_x, padded_center_y = padded_n_x//2, padded_n_y//2
    # padding margins
    left_m_x, right_m_x = (
        padded_center_x-n_x//2,
        padded_center_x+n_x//2
    )
    left_m_y, right_m_y = (
        padded_center_y-n_y//2,
        padded_center_y+n_y//2
    )

    # FFT over time
    # Find the smallest 2**k >= n_samples
    n_samples = 1 << (n_samples-1).bit_length()
    rf_ft_t = np.fft.fft(rf, n_samples, axis=-1)
    rf_ft_t = np.fft.fftshift(rf_ft_t)

    dt = 1./acq_params.sampling_frequency
    freq = np.fft.fftfreq(n_samples, d=dt)
    freq = np.fft.fftshift(freq)

    # FFT over OX, OY
    rf_ft = np.zeros(shape=(padded_n_x, padded_n_y, n_samples),dtype=np.complex64)
    for t in range(n_samples):
        tmp = np.zeros(shape=(padded_n_x, padded_n_y), dtype=np.complex64)
        tmp[left_m_x:right_m_x, left_m_y: right_m_y] = rf_ft_t[:, :, t]
        tmp = np.fft.fft2(tmp)
        tmp = np.fft.fftshift(tmp)
        rf_ft[:, :, t] = tmp

    # Interpolate.
    dx, dy = sys_params.probe.pitch, sys_params.probe.pitch
    dkx = 2*np.pi/(padded_n_x*dx)
    dky = 2*np.pi/(padded_n_y*dy)
    kx = np.arange(-padded_n_x/2+1, padded_n_x/2+1)*dkx
    ky = np.arange(-padded_n_y/2+1, padded_n_y/2+1)*dky

    f_max = freq[n_samples-1]
    kz_max = 2*np.pi*f_max/acq_params.speed_of_sound
    kz = np.linspace(0.0, kz_max, num=n_z)

    rf_ft_interp = np.zeros(
        shape=(padded_n_x, padded_n_y, n_z),
        dtype=np.complex64
    )
    for x, y in zip(range(padded_n_x), range(padded_n_y)):
        ft_line = rf_ft[x, y, :]
        samples = acq_params.speed_of_sound/(4*np.pi) \
                  * ((kz**2+kx[x]**2+ky[y]**2) / kz)
        w       = acq_params.speed_of_sound/(4*np.pi) \
                  * ((kz**2-kx[x]**2-ky[y]**2) / kz)
        samples[0] = 0
        w[0] = 0
        ft_line_interp = np.interp(samples, freq, ft_line, left=0, right=0)
        rf_ft_interp[x, y, :] = w*ft_line_interp

    # IFFT over OZ
    result = np.fft.ifft(rf_ft_interp, axis=-1)
    for z in range(n_z):
        tmp = np.fft.ifft2(result[:, :, z])
        result[:, :, z] = tmp
    return result[left_m_x:right_m_x, left_m_y: right_m_y, :]


if __name__ == "__main__":
    rf, acq_params, sys_params = load_data(
        "/home/pjarosik/data/us4us/hsdi/data/field/scattered_field_3D_biala_real_TXRX4x4.mat"
    )
    volume = reconstruct_hri(rf, acq_params, sys_params, n_z=1024)
    n_x, n_y, n_z = volume.shape
    print(volume.shape)
    volume = np.abs(volume)
    v_max = np.max(volume)
    volume = volume / v_max

    # plot two selected planes of the volume
    plane1 = volume[n_x//2, :, :].T
    plane2 = volume[:, n_y//2, :].T

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(
        np.clip(
            20*np.log10(plane1/np.max(plane1)),
            a_min=-40,
            a_max=0
        ),
        cmap='gray'
    )
    ax2.imshow(
        np.clip(
            20*np.log10(plane2/np.max(plane2)),
            a_min=-40,
            a_max=0
        ),
        cmap='gray')
    plt.show()



