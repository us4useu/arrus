import functools
import numpy as np

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
