import functools
import numpy as np
import re
import dataclasses

# ------------------------------------------ Assertions
def assert_true(value: bool, desc: str = ""):
    if not value:
        msg = "Validation error"
        if desc:
            msg += ": %s." % desc
        raise ValueError(msg)


def assert_not_none(values):
    none_values = [(k, v) for k, v in values if v is None]
    if len(none_values) > 0:
        none_parameters = [k for k, _ in values]
        none_parameters = ", ".join(none_parameters)
        msg = "Validation error: following values are required: %s" % (
            none_parameters
        )
        raise ValueError(msg)


def _assert_equal(expected, actual, desc: str = ""):
    if expected != actual:
        msg = "Validation error"
        if desc:
            msg += ": %s." % desc
        raise ValueError(msg)

# ------------------------------------------ Others
def create_aligned_array(shape, dtype, alignment):
    dtype = np.dtype(dtype)
    size = functools.reduce(lambda a, b: a*b, shape, 1)
    nbytes = size*dtype.itemsize
    buffer = np.empty(nbytes+alignment, dtype=np.uint8)
    if buffer.ctypes.data % alignment != 0:
        offset = -buffer.ctypes.data % alignment
        buffer = buffer[offset:(offset+nbytes)].view(dtype).reshape(*shape)
    assert buffer.ctypes.data % alignment == 0
    return buffer


def convert_camel_to_snake_case(string: str):
    """
    Converts a string written in camelCase to a string written in a snake_case.

    :param string: a camelCase string to convert
    :return: a string converted to snake_case
    """
    words = re.findall("[a-zA-Z][^A-Z]*", string)
    words = (w.lower() for w in words)
    return "_".join(words)


def convert_snake_to_camel_case(string: str):
    """
    Converts a string written in snake_case to a string written in a camelCase.

    The output string will start with lower case letter.

    :param string: a snake_case string to convert
    :return: a string converted to camelCase
    """
    words = string.split("_")
    capitalized_words = (w.capitalize() for w in words[1:])
    first_word = words[0]
    first_word = first_word[:1].lower() + first_word[1:]
    words = [first_word]
    words.extend(capitalized_words)
    return "".join(words)


