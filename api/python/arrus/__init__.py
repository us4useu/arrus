"""ARRUS."""

import os
# use arrus-core.dll
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.join(os.path.abspath(__file__)))


__version__ = "${PROJECT_VERSION}"

from arrus.session import Session
from arrus.logging import (
    set_clog_level,
    add_log_file
)

from arrus.exceptions import *


