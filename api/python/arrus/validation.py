from collections.abc import Iterable


def assert_not_none(value, parameter_name):
    if value is None:
        raise InvalidParameterError(parameter_name, "should be not None")


def assert_none(value, parameter_name):
    if value is not None:
        raise InvalidParameterError(parameter_name, "should not be provided")


def assert_shape(array, expected_shape, parameter_name, strict=False):
    if strict:
        shape = array.shape
    else:
        shape = array.flatten().shape
    if shape != expected_shape:
        raise InvalidParameterError(parameter_name, "expected shape: %s " %
                                    expected_shape)


def assert_not_greater_than(value, maximum, parameter_name):
    if value > maximum:
        raise InvalidParameterError(parameter_name,
                                    "should not be greater than %d" % maximum)


def assert_in_range(actual, expected, parameter_name):
    if not isinstance(actual, Iterable):
        a_start, a_end = actual, actual
    else:
        a_start, a_end = actual
    e_start, e_end = expected
    if not(a_start >= e_start and a_end <= e_end):
        raise InvalidParameterError(parameter_name,
                                    "%s expected in range %s" %
                                    (str(actual), str(expected)))


def assert_one_of(value, collection, parameter_name):
    """Note! this function is only for exact comparing."""
    assert isinstance(collection, Iterable)
    s = set(collection)
    if value not in s:
        raise InvalidParameterError(parameter_name, "should be one of: %s" % str(s))


def assert_non_negative(value, parameter_name):
    if value < 0:
        raise InvalidParameterError(parameter_name, "should be non-negative")


def assert_positive(value, parameter_name):
    if value <= 0:
        raise InvalidParameterError(parameter_name, "should be positive")


def assert_type(o, t, parameter_name):
    if not isinstance(o, t):
        raise InvalidParameterError(parameter_name,
                                    "should be of type: %s" % str(t))


class InvalidParameterError(ValueError):
    MSG_PATTERN = "Invalid parameter '%s': %s"

    def __init__(self, param, msg):
        super().__init__(InvalidParameterError.MSG_PATTERN % (param, msg))



