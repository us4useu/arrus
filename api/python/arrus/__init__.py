"""ARRUS."""
import logging
from logging import ERROR, WARNING, INFO, DEBUG

_logger = logging.getLogger(__package__)
DEFAULT_LOGGER_LEVEL = logging.INFO
_logger.setLevel(DEFAULT_LOGGER_LEVEL)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(DEFAULT_LOGGER_LEVEL)
_logger_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_console_handler.setFormatter(_logger_formatter)
_logger.addHandler(_console_handler)


def set_log_level(level):
    """
    Sets logger level.

    Available levels ERROR, WARNING, INFO, DEBUG
    """
    _logger.setLevel(level)
    _console_handler.setLevel(level)


def add_log_file(filename, level):
    log_file_handler = logging.FileHandler(filename)
    log_file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_file_handler.setFormatter(file_formatter)
    _logger.addHandler(log_file_handler)


# TODO temporary ommiting importing some of the modules here, when
# low-level API is not available (for example currently on Unix systems).
import importlib
import importlib.util
is_ius4oem = importlib.util.find_spec("arrus.devices._ius4oem")
is_ihv256 = importlib.util.find_spec("arrus.devices._ihv256")
is_idbarlite = importlib.util.find_spec("arrus.devices._idbarlite")
if is_ius4oem and is_ihv256 and is_idbarlite:
    # Legacy support attributes.
    import arrus.devices.device as device
    import arrus.session as session
    import arrus.interface as interface
    import arrus.beam as beam
    from arrus.session import Session
else:
    _logger.warn("Low-level API libraries are currently not available, "
                 "providing minimal version of the package.")

from arrus.params import *
import arrus.ops as ops