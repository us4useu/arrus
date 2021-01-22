"""
Logging tools for arrus.
Currently the arrus logging mechanism is
implemented in c++ using boost::log library.
"""
import arrus.core

# Wrap arrus core logging levels.
TRACE = arrus.core.LogSeverity_TRACE
DEBUG = arrus.core.LogSeverity_DEBUG
INFO = arrus.core.LogSeverity_INFO
WARNING = arrus.core.LogSeverity_WARNING
ERROR = arrus.core.LogSeverity_ERROR
FATAL = arrus.core.LogSeverity_FATAL

DEFAULT_LEVEL = INFO

# Init default level logging.
arrus.core.initLoggingMechanism(DEFAULT_LEVEL)


def set_clog_level(level):
    """
    Sets console log level output.

    :param level: log level to use
    """
    return arrus.core.setClogLevel(level)


def add_log_file(filepath, level):
    """
    Adds message logging to given file.

    :param filepath: path to output log file
    :param level: severity level
    """
    return arrus.core.addLogFile(filepath, level)


def get_logger():
    """
    Returns a handle to new logger.

    :return:
    """
    return arrus.core.getLogger()


DEFAULT_LOGGER = get_logger()


def log(level, msg):
    DEFAULT_LOGGER.log(level, msg)
