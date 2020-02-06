"""Arius SDK."""

import logging

_logger = logging.getLogger(__package__)
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(logger_formatter)
    _logger.addHandler(console_handler)

from arius.python import *
import arius.python.arius.devices.device as device
import arius.python.arius.beam as beam
