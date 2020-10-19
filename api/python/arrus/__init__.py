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
    Sets logging level.

    :param level: logging level to set, available levels ``ERROR``, \
        ``WARNING``, ``INFO``, ``DEBUG``
    """
    _logger.setLevel(level)
    _console_handler.setLevel(level)


def add_log_file(filename: str, level):
    """
    Add file, where logging information should appear.

    :param filename: a path to the output file
    :param level: level to set, available levels: ``ERROR``, \
        ``WARNING``, ``INFO``, ``DEBUG``
    """
    log_file_handler = logging.FileHandler(filename)
    log_file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_file_handler.setFormatter(file_formatter)
    _logger.addHandler(log_file_handler)


import importlib
import importlib.util
is_ius4oem = importlib.util.find_spec("arrus.devices._ius4oem")
is_ihv256 = importlib.util.find_spec("arrus.devices._ihv256")
is_idbarlite = importlib.util.find_spec("arrus.devices._idbarlite")
if is_ius4oem and is_ihv256 and is_idbarlite:
    # Legacy support attributes.
    import arrus.devices.device as device
    import arrus.session as session
    import arrus.beam as beam
    from arrus.session import Session
    from arrus.session import SessionCfg
    from arrus.devices.us4oem import Us4OEMCfg
    from arrus.devices.us4oem import ChannelMapping
    from arrus.system import CustomUs4RCfg
else:
    _logger.warn("Low-level API libraries are currently not available, "
                 "providing mocked version of the package.")

from arrus.params import *
import arrus.ops as ops