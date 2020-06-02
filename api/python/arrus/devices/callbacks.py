import arrus.devices.ius4oem as _ius4oem
from logging import DEBUG, INFO
import logging
_logger = logging.getLogger(__name__)

class ScheduleReceiveCallback(_ius4oem.ScheduleReceiveCallback):
    def __init__(self, callback_fn):
        super().__init__()
        self.callback_fn = callback_fn

    def run(self, event):
        try:
            self.callback_fn(event)
        except Exception as e:
            _logger.exception(e)
