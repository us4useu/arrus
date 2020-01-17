from typing import get_type_hints
import numpy as np


class SystemParameters:
    n_elements: int
    pitch: float


class Tx:
    frequency: float
    n_periods: int
    angles: np.ndarray
    focus: float
    aperture_size: int


class Rx:
    sampling_frequency: float
    aperture_size: int


class AcquisitionParameters:
    mode: str
    speed_of_sound: float
    tx: Tx
    rx: Rx


MATLAB_ROOT_STRUCTURES = {
    "systemParameters" : SystemParameters,
    "acquisitionParameters": AcquisitionParameters
}


def loadmat(filename: str):
    """
    
    :param filename: 
    :return: a tupple: system_parameters, acquisition_parameters
    """
    pass



