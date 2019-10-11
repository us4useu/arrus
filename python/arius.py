""" ARIUS API. """
import numpy as np

_DEVICES = {
    "MOCKHAL": lambda kwargs: MockHAL(**kwargs)
}


def getAvailableDevices():
    """
    Returns a list of available devices (HAL intances).
    """
    return [Device(name) for name in _DEVICES.keys()]


def getHALInstance(name, **kwargs):
    """
    Returns new instance of HAL connector to the device.

    :param name: name of the HAL device.
    :return: HAL instance
    """
    return _DEVICES[name](kwargs)


class MockHAL:
    def __init__(self, data=None):
        """
        MockHAL's constructor can take one optional argument 'data':
        - 3-D matrix with dimensions: (x, y, z) - single 'RF frame'
        - 4-D matrix with dimensions: (x, y, z, n) - collection of RF
          frames to cycle through; the last dimension corresponds to
          the RF frame number.

        When no argument is provided: three random RF frames
        (32, 512, 32) are generated.

        :param data: input data to return
        """
        if data is None:
            self.data = np.random.rand(32, 512, 32, 4)
        else:
            shape = data.shape
            if len(shape) == 3:
                self.data = data.reshape(shape + (1,))
            elif len(shape == 4):
                self.data = data
            else:
                raise ValueError("Unsupported input data shape, "
                                 "should be 3-D or 4-D.")
        self.frameIdx = 0
        self.isConfigured = False
        self.isStarted = False

    def configure(self, json):
        """
        Applies TX/RX configuration stored in given JSON string.

        :param json: configuration string
        :return None
        """
        if self.isStarted:
            raise ValueError("Device cannot be configured now. "
                             "Call 'stop' first.")
        self.isConfigured = True
        print("Loaded configuration file.")

    def start(self):
        """
        Starts the device.
        """
        if not self.isConfigured:
            raise ValueError("Device is not configured. "
                             "Call 'configure' first.")
        self.isStarted = True
        print("Device started.")

    def stop(self):
        """
        Stops the device.
        """
        self._assertIsStarted()
        self.isStarted = False
        print("Device is stopped.")

    def sync(self, frameIdx):
        """
        Sync. for the next acquisition.
        """
        # TODO(pjarosik) what is the purpose of the frameIdx parameter?
        self._assertIsStarted()
        nframes = self.data.shape[-1]
        self.frameIdx = (self.frameIdx + 1) % nframes

    def getData(self):
        """
        Returns current data buffer.

        Output dimensions: ESC (Event, Sample, Channel).

        :return: numpy array, metadata
        """
        self._assertIsStarted()
        return (
            self.data[:, :, :, self.frameIdx].squeeze(),
            Metadata(self.frameIdx)
        )

    def _assertIsStarted(self):
        if not self.isStarted:
            raise ValueError("Device is not started. Call 'start' first.")


class Metadata:
    def __init__(self, frameIdx):
        self.frameIdx = frameIdx


class Device:
    def __init__(self, name):
        self.name = name
