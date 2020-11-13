import arrus.ops.imaging
import arrus.exceptions
from .imaging import (
    create_lin_sequence
)


def _identity_func(context):
    return context.op


# TODO should depend on the device, currently us4r is supported only
_kernel_registry = {
    arrus.ops.us4r.TxRxSequence: _identity_func,
    arrus.ops.imaging.LinSequence: create_lin_sequence
}


def get_kernel(op_type):
    if op_type not in _kernel_registry:
        raise arrus.exceptions.IllegalArgumentError(
            f"Operation {op_type} is not supported.")
    return _kernel_registry[op_type]