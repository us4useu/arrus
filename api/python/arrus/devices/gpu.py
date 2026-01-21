from typing import Optional

from arrus.devices.device import Device, DeviceId, DeviceType

DEVICE_TYPE = DeviceType("GPU")


class GpuSettings:
    """
    GPU settings provided by the user
    (in the session settings a.k.a prototxt file).
    """

    def __init__(self, handle):
        self._handle = handle

    @property
    def memory_limit_percentage(self) -> Optional[float]:
        return self._handle.getMemoryLimitPercentage()

    @property
    def use_memory_pool(self) -> bool:
        return self._handle.getUseMemoryPool()


class Gpu(Device):
    """
    GPU device.

    NOTE: the current implementation of this class plays only the role
    of GpuSettings handler. The only functionality this class exposes right
    now is the access to the GPU settings specified by user the session settings.
    """

    def __init__(self, handle):
        super().__init__()
        self._handle = handle
        self._device_id = DeviceId(
            device_type=DEVICE_TYPE,
            ordinal=self._handle.getDeviceId().getOrdinal()
        )

    def get_device_id(self):
        return self._device_id

    def get_settings(self):
        return GpuSettings(self._handle.getSettings())



