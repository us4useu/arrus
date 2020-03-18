from typing import get_type_hints
import numpy as np
import scipy.io
import python.utils
from dataclasses import dataclass
import inspect

class BmodeDescriptor:
    """An abstract class for a b-mode acquisition descriptor."""
    pass

@dataclass(frozen=True)
class SystemParameters(BmodeDescriptor):
    """
    All parameters related to the system-wide configuration.

    :var n_elements: number of probe's elements [elements]
    :var pitch: distance between elements of the probe [m]
    """
    n_elements: np.uint16
    pitch: np.float64


@dataclass(frozen=True)
class Tx(BmodeDescriptor):
    """
    A description of the signal transmission.

    :var frequency: transmitted carrier/nominal/center frequency [Hz]
    :var n_periods: number of sine periods in the transmitted burst
    :var angles: array of transmission angles [rad]
    :var focus: focal depth [m]
    :var aperture_size: size of a transmit aperture [elements]
    """
    frequency: np.float64
    n_periods: np.uint16
    angles: list
    focus: np.float64
    aperture_size: np.uint16


@dataclass(frozen=True)
class Rx(BmodeDescriptor):
    """
    A description of the signal reception.

    :var sampling_frequency: signal sampling frequency [Hz]
    :var aperture_size: size of rx aperture [elements]
    """
    sampling_frequency: np.float64
    aperture_size: np.uint16


@dataclass(frozen=True)
class AcquisitionParameters(BmodeDescriptor):
    """
    All parameters related to an acquisition of the given RF frame.

    :var mode: transmission scheme type {‘sta’,’pwi’,’lin’}
    :var speed_of_sound: assumed speed of sound [m/s]
    :var tx: description of the signal transmission
    :var rx: description of the signal reception
    """
    mode: str
    speed_of_sound: np.float64
    tx: Tx
    rx: Rx


MATLAB_ROOT_STRUCTURES = [
    ("systemParameters", SystemParameters),
    ("acquisitionParameters", AcquisitionParameters)
]


MATLAB_ROOT_STRUCTURES_DICT = dict(MATLAB_ROOT_STRUCTURES)


def load_matlab_file(mat_file):
    """
    Reads a given MATLAB file and returns RF data and pythonized structures
    describing how the data have been acquired.

    Currently, this function returns a tuple:
    - a :class:`numpy.ndarray` which contains RF data,
    - an instance of :class:`.SystemParameters`,
    - an instance of :class:`.AcquisitionParameters`.

    Each empty MATLAB array value ([]) will be replaced with 'None' value.

    :param mat_file: a file to load; if str,
        a path to given mat_file, the data will be loaded
        using :func:`scipy.io.loadmat`,
        otherwise expects matlab workspace as an input
    :return: a tuple: rf, system_parameters, acquisition_parameters
    """
    if type(mat_file) == str:
        mat_file = scipy.io.loadmat(mat_file)
    rf = mat_file['rf']
    structures = tuple(_load_matlab_structure(attr_class, mat_file[attr_name])
                 for attr_name, attr_class in MATLAB_ROOT_STRUCTURES)
    return (rf,) + structures


def save_matlab_file(path, rf, system_parameters, acquisition_parameters):
    """
    Saves a given RF data frame and all related structures to a .mat file
    located in a given `path`.

    This function saves 'None' values as an empty matlab array ([]).

    :param path: a location with filename where to save a given matlab file
    :param rf: an RF data frame to save
    :param system_parameters: a :class:`.SystemParameters` of a given RF frame
    :param acquisition_parameters: a :class:`.AcquisitionParameters` of a given RF frame
    """
    result = {}
    func_frame = inspect.currentframe()
    args, _, _, vals = inspect.getargvalues(func_frame)
    args = [(arg, vals[arg]) for arg in args[1:]]

    matlab_keys_camel_cased = [
        python.utils.convert_camel_to_snake_case(k)
        for k, _ in MATLAB_ROOT_STRUCTURES
    ]
    for key, value in args:
        if key == 'rf':
            result[key] = value
            continue
        if key not in matlab_keys_camel_cased:
            raise ValueError(
                "Unrecognized parameter: '%s' should be one of: '%s'."
                    % (str(key), str(matlab_keys_camel_cased))
            )
        matlab_key = python.utils.convert_snake_to_camel_case(key)
        expected_class = MATLAB_ROOT_STRUCTURES_DICT[matlab_key]
        if expected_class != value.__class__:
            raise ValueError(
                "Expected '%s' instead of '%s' for '%s'."
                    % (str(expected_class), str(value.__class__), key)
            )
        result[matlab_key] = _convert_to_matlab_structure(value)
    scipy.io.savemat(path, result)


def _load_matlab_structure(py_class, mat_structure):
    py_class_attrs = get_type_hints(py_class)
    py_constructor_kwargs = {}
    for attr, attr_class in py_class_attrs.items():
        attr_mat_name = python.utils.convert_snake_to_camel_case(attr)
        if not attr_mat_name in mat_structure.dtype.names:
            raise ValueError(
                "Given MATLAB data does not contain '%s' field." % attr_mat_name
            )
        mat_value = mat_structure[attr_mat_name][0][0]
        if mat_value.size == 0:
            value = None
        elif issubclass(attr_class, BmodeDescriptor):
            value = _load_matlab_structure(attr_class, mat_value)
        elif attr_class == np.ndarray:
            value = mat_value.squeeze()
        elif attr_class == list:
            value = mat_value.squeeze().tolist()
        elif attr_class == str:
            value = str(mat_value[0])
        else:
            # Scalar.
            mat_value_shape = mat_value.squeeze().shape
            if (len(mat_value_shape) > 1 or
               (len(mat_value_shape) == 1 and mat_value_shape[0] > 1)):
                raise ValueError(
                    "Value for '%s' should be a scalar." % attr)
            value = mat_value[0][0]
            if type(value) != attr_class:
                if (np.can_cast(type(value), attr_class, casting="safe")
                    or np.can_cast(value, attr_class, casting="safe")):
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
        py_constructor_kwargs[attr] = value
    return py_class(**py_constructor_kwargs)


def _convert_to_matlab_structure(structure):
    result = {}
    # Validate input structure.
    cls = structure.__class__
    py_class_attrs = get_type_hints(cls)
    structure_fields = dict(vars(structure))
    if structure_fields.keys() != py_class_attrs.keys():
        raise ValueError(
            "A structure of class '%s' should have exactly fields: %s. "
                % (cls, str(list(py_class_attrs.keys())))
        )
    for attr, expected_cls in py_class_attrs.items():
        value = structure_fields[attr]
        # TODO(pjarosik) consider marking an attr as an optional value somehow
        value_type = type(value)
        if value is None:
            value = []
        elif issubclass(expected_cls, BmodeDescriptor):
            _assert_is_type_of(value, expected_cls, attr)
            value = _convert_to_matlab_structure(value)
        elif expected_cls == list:
            _assert_is_type_of(value, expected_cls, attr)
        elif expected_cls == np.ndarray:
            if value_type == list:
                value = np.array(value)
            else:
                _assert_is_type_of(value, expected_cls, attr)

        # The first part of the below condition is to handle str correctly.
        elif (np.can_cast(value_type, expected_cls, casting='safe') or
              np.can_cast(value, expected_cls, casting="safe")):
            value = expected_cls(value)
        else:
            raise ValueError(
                "Unsupported value type '%s' for '%s'."
                    %(type(value), attr)
            )
        result[python.utils.convert_snake_to_camel_case(attr)] = value
    return result


def _assert_is_type_of(value, cls, attr):
    if not isinstance(value, cls):
        raise ValueError(
            "A value for '%s' should be of type: %s, but is '%s'."
            % (attr, str(cls), str(type(value)))
        )


