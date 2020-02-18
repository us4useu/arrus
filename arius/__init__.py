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
import arius.python.devices.device as device
import arius.python.session as session
import arius.python.interface as interface
import arius.python.beam as beam
