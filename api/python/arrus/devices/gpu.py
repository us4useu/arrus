from arrus.devices.device import Device


class GPU(Device):

    def __init__(self, index: int):
        super().__init__("CPU", index)