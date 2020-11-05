import dataclasses
import arrus.devices.us4r
import arrus.medium
import arrus.ops

@dataclasses.dataclass(frozen=True)
class KernelExecutionContext:
    """
    Kernel execution context.

    This function contains all data that is available for the implementation of the kernel.

    :param device: device on which the kernel will be executed
    :param medium: the medium assumed for the current session
    :param op: operation to perform
    :param custom: custom data
    """
    device: arrus.devices.us4r.Us4RDTO
    medium: arrus.medium.MediumDTO
    op: arrus.ops.Operation
    custom: dict
