from arrus.devices.device import Device, DeviceId, DeviceType

DEVICE_TYPE = DeviceType("GPU")


class GPU(Device):
    def __init__(self, index):
        super().__init__()
        self._index = index

    def get_device_id(self):
        return DeviceId(DEVICE_TYPE, self._index)

