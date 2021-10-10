import arrus.ops.imaging
import arrus.exceptions
from .imaging import (
    process_simple_tx_rx_sequence
)


def _identity_func(context):
    return context.op


# TODO should depend on the device, currently us4r is supported only
_kernel_registry = {
    arrus.ops.us4r.TxRxSequence: _identity_func,
    arrus.ops.imaging.LinSequence: process_simple_tx_rx_sequence,
    arrus.ops.imaging.PwiSequence: process_simple_tx_rx_sequence,
    arrus.ops.imaging.StaSequence: process_simple_tx_rx_sequence
}


def get_kernel(op_type):
    if op_type not in _kernel_registry:
        raise arrus.exceptions.IllegalArgumentError(
            f"Operation {op_type} is not supported.")
    return _kernel_registry[op_type]


def register_kernel(op_type, func):
    if op_type in _kernel_registry:
        raise arrus.exceptions.IllegalArgumentError(f"Kernel for op {op_type} "
                                                    f"already registered")
    _kernel_registry[op_type] = func