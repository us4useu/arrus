"""
Definition of exceptions thrown by arrus functions.
"""

class ArrusError(Exception):
    pass


class IllegalArgumentError(ArrusError, ValueError):
    pass


class DeviceNotFoundError(ArrusError, ValueError):
    pass


class IllegalStateError(ArrusError, RuntimeError):
    pass


class TimeoutError(ArrusError, TimeoutError):
    pass