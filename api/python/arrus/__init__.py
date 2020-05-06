"""ARRUS."""
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
else:
    _logger.warn("Low-level API libraries are currently not available, "
                 "providing minimal version of the package.")

