import arius.python.devices.iarius as _iarius

class ScheduleReceiveCallback(_iarius.ScheduleReceiveCallback):

    def __init__(self, callback_fn):
        super().__init__()
        self.callback_fn = callback_fn

    def run(self, event):
        self.callback_fn(event)
