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

# TODO temporary ommiting importing some of the modules here, when
# low-level API is not available (for example currently on Unix systems).
import importlib.util
is_iarius = importlib.util.find_spec("arius.devices._iarius")
is_ihv256 = importlib.util.find_spec("arius.devices._ihv256")
is_idbarlite = importlib.util.find_spec("arius.devices._idbarlite")
if is_iarius and is_ihv256 and is_idbarlite:
    # Legacy support attributes.
    import arius.devices.device as device
    import arius.session as session
    import arius.interface as interface
    import arius.beam as beam
else:
    _logger.warn("Low-level API libraries are currently not available, "
                 "providing minimal version of the package.")

