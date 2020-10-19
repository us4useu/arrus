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


# Legacy support attributes.
import arrus.devices.device as device
import arrus.session as session
import arrus.beam as beam
from arrus.session import MockSession

from arrus.params import *
import arrus.ops as ops