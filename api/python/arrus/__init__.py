"""ARRUS."""

import os
# use arrus-core.dll from the python package
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.join(os.path.abspath(__file__)))

__version__ = "${ARRUS_PROJECT_VERSION}"

# Exceptions
from arrus.exceptions import *

# The below depends on arrus.core wrapper.
# TODO: currently this is only for the convenience when writing unit tests
# TODO eventually we will need to remove the below
# TODO: for arrus.processing implementations tests
import importlib.util
arrus_core_module_exists = importlib.util.find_spec("arrus.core") is not None
# Logging

if arrus_core_module_exists:

    from arrus.logging import (
        set_clog_level,
        add_log_file
    )
    # Session
    from arrus.session import Session

else:
    print("Warning: running without arrus.core module.")


