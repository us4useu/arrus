from dataclasses import dataclass
import numpy as np
import scipy.io
import re
import pathlib

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
    :return: (rf, acquisition_parameters, system_parameters
    """
    rf = scipy.io.loadmat(path)['Matrix']
    filename = pathlib.Path(path).name
    dims = re.findall(r'TXRX(\d+)x(\d+)\.mat', filename)
    if len(dims) > 0 and len(dims[0]) != 2:
        raise ValueError("Input file should be of format: \w+TXRX\d+x\d+")
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



