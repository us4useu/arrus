import dataclasses
import arrus.medium
import arrus.ops
import arrus.devices.device

@dataclasses.dataclass(frozen=True)
class KernelExecutionContext:
    """
    Kernel execution context.

    This function contains all data that is available for the implementation of
    the kernel.

    :param device: device on which the kernel will be executed
    :param medium: the medium assumed for the current session
    :param op: operation to perform
    :param custom: custom data
    """
    device: arrus.devices.device.UltrasoundDeviceDTO
    medium: arrus.medium.MediumDTO
    op: arrus.ops.Operation
    custom: dict
