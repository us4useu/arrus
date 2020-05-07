import arrus.devices.ius4oem as _ius4oem

class ScheduleReceiveCallback(_ius4oem.ScheduleReceiveCallback):
    def __init__(self, callback_fn):
        super().__init__()
        self.callback_fn = callback_fn

    def run(self, event):
        self.callback_fn(event)
