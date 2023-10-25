import dataclasses
from typing import Any, List

import arrus.medium
import arrus.ops
import arrus.ops.us4r
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
    device: arrus.devices.ultrasound.UltrasoundDTO
    medium: arrus.medium.MediumDTO
    op: arrus.ops.Operation
    custom: dict
    hardware_ddc: arrus.ops.us4r.DigitalDownConversion = None
    constants: List[arrus.framework.Constant] = tuple()


@dataclasses.dataclass(frozen=True)
class ConversionResults:
    sequence: object
    constants: List[Any]

