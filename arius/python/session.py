class Session:
    def __init__(self):
        self._devices = self._detect_devices()

    def _detect_devices(self):
        return []


class InteractiveSession(Session):
    def __init__(self):
        super().__init__()


    def get_device(self, name: str):
        pass


