from typing import get_type_hints
import numpy as np
import scipy.io
import python.utils

class BmodeDescriptor:
    pass

class SystemParameters(BmodeDescriptor):
    n_elements: np.uint16
    pitch: np.float64


class Tx(BmodeDescriptor):
    frequency: np.float64
    n_periods: np.unit16
    angles: np.ndarray
    focus: np.float64
    aperture_size: np.uint16


class Rx(BmodeDescriptor):
    sampling_frequency: np.float64
    aperture_size: np.uint16


class AcquisitionParameters(BmodeDescriptor):
    mode: str
    speed_of_sound: np.float64
    tx: Tx
    rx: Rx


MATLAB_ROOT_STRUCTURES = [
    ("systemParameters", SystemParameters),
    ("acquisitionParameters", AcquisitionParameters)
]


def load_matlab_file(mat_file):
    """
    
    :param mat_file: a file to load; if str,
        a path to given mat_file, will be loaded using scipy.io.loadmat,
        otherwise expects matlab structure
    :return: a tupple: system_parameters, acquisition_parameters
    """
    if type(mat_file) == str:
        mat_file = scipy.io.loadmat(mat_file)
    return tuple(_load_matlab_structure(attr_class, mat_file[attr_name])
                 for attr_name, attr_class in MATLAB_ROOT_STRUCTURES)


def _load_matlab_structure(py_class, mat_structure):
    py_class_attrs = get_type_hints(py_class)
    py_class_instance = py_class()
    for attr, attr_class in py_class_attrs.items():
        attr_mat_name = python.utils.convert_snake_to_camel_case(attr)
        if not attr_mat_name in mat_structure:
            raise ValueError(
                "Given MATLAB data does not contain '%s' field." % attr_mat_name
            )
        mat_array = mat_structure[attr_mat_name]
        mat_value = mat_structure[attr_mat_name][0][0]
        if mat_array.size == 0:
            value = None
        if issubclass(attr_class, BmodeDescriptor):
            value = _load_matlab_structure(attr_class, mat_value)
        elif attr_class == np.ndarray:
            value = mat_value.squeeze()
        else:
            # Scalar.
            mat_value_shape = mat_value.squeeze().shape
            if len(mat_value_shape) > 1 or mat_value_shape[0] > 1:
                raise ValueError("Value for '%s' should be a scalar.")
            value = mat_value[0][0]
            if type(value) != attr_class:
                if np.can_cast(type(value), attr_class, casting="safe"):
                    value = attr_class(value)
                else:
                    raise ValueError(
                    "Value for '%s': can't be cast from '%s', to '%s'." %
                        (attr, type(value), attr_class)
                    )
        if value is not None and type(value) != attr_class:
            raise ValueError(
                "Invalid value for '%s': should be '%s', is '%s'." %
                    (attr, attr_class, type(value))
            )
        setattr(py_class_instance, attr, value)
    return py_class_instance


