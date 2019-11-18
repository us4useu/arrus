""" ARIUS API. """
from .beam import BeamShape


class InteractiveSession:
    def __init__(self):
        # TODO(pjarosik) detect available devices
        pass

    def get_device(self, name: str):
        pass


class Probe:
    def __init__(self):
        pass

    def transmit_and_record(self, **kwargs):
        # TODO(pjarosik) if beam_shape is provided, use it
        # otherwise, use given delay_profile
        pass


class DelayProfile:
    pass
