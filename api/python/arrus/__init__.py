"""ARRUS."""

import os
# use arrus-core.dll from the python package
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.join(os.path.abspath(__file__)))

# Logging
from arrus.logging import (
    set_clog_level,
    add_log_file
)

# Exceptions
from arrus.exceptions import *

# Session
from arrus.session import Session


