from arrus.devices.device import Device


class CPU(Device):

    def __init__(self, index: int):
        super().__init__("CPU", index)