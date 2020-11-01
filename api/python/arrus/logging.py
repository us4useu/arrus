"""
Logging tools for arrus.
Currently the arrus logging mechanism is
implemented in c++ using boost::log library.
"""
import arrus.core

# Wrap arrus core logging levels.
TRACE = arrus.core.LogSeverity_trace
DEBUG = arrus.core.LogSeverity_debug
INFO = arrus.core.LogSeverity_info
WARNING = arrus.core.LogSeverity_warning
ERROR = arrus.core.LogSeverity_error
FATAL = arrus.core.LogSeverity_fatal

DEFAULT_LEVEL = INFO

# Init default level logging.
arrus.core.init_logging_mechanism(DEFAULT_LEVEL)


def set_clog_level(level):
    return arrus.core.set_clog_level(level)


def add_log_file(filepath, level):
    """
    Adds logging to given file.

    :param filepath: path to output log file
    :param level: severity level
    """
    return arrus.core.add_log_file(filepath, level)


def get_logger():
    """
    Returns a handle to new logger.

    :return:
    """
    return arrus.core.get_logger()


DEFAULT_LOGGER = get_logger()


def log(level, msg):
    DEFAULT_LOGGER.log(level, msg)
