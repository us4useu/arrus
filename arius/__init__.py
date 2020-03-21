"""Arius SDK."""

import logging

_logger = logging.getLogger(__package__)
LOGGER_LEVEL = logging.INFO
_logger.setLevel(LOGGER_LEVEL)

if not _logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOGGER_LEVEL)
    logger_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(logger_formatter)
    _logger.addHandler(console_handler)

from arius.python import *

# TODO temporary ommiting importing some of the modules here, when
# low-level API is not available (for example currently on Unix systems).
import importlib

is_iarius = importlib.util.find_spec("arius.python.devices._iarius")

if is_iarius:
    import arius.python.devices.device as device
    import arius.python.session as session
    import arius.python.interface as interface
    import arius.python.beam as beam
else:
    _logger.warn("Low-level API libraries are not available, "
                 "providing minimal version of the package.")

x = 2512
import arius.python.utils as utils
