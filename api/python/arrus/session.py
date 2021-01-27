import abc
import numpy as np
import importlib
import importlib.util
import dataclasses

import arrus.core
import arrus.exceptions
import arrus.devices.us4r
import arrus.devices.mock_us4r
import arrus.medium
import arrus.metadata
import arrus.params
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.ops.us4r
import arrus.ops.imaging


class AbstractSession(abc.ABC):
    """
    An abstract class of session.

    This class is not intended to be instantiated.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_device(self, id: str):
        """
        Returns a device located at given path.

        :param id: a path to a device, for example '/Us4R:0'
        :return: a device located in a given path.
        """
        raise ValueError("Tried to access an abstract method.")

    @abc.abstractmethod
    def set_current_medium(self, medium: arrus.medium.Medium):
        """
        Sets a medium in the current session context.

        :param medium: medium description to set
        """
        raise ValueError("Tried to access an abstract method.")


@dataclasses.dataclass(frozen=True)
class SessionContext:
    medium: arrus.medium.Medium


class Session(AbstractSession):
    """
    A communication session with the ultrasound system.
    Currently, only localhost session is available.
    """

    def __init__(self, cfg_path: str = None,
                 medium: arrus.medium.Medium = None):
        """
        Session constructor.

        :param cfg_path: a path to configuration file
        :param medium: medium description to set in context
        """
        super().__init__()
        self._session_handle = arrus.core.createSessionSharedHandle(cfg_path)
        self._context = SessionContext(medium=medium)
        self._py_devices = self._create_py_devices()

    def upload(self, scheme: arrus.ops.us4r.Scheme):
        """
        
        :param scheme:
        :return:
        """

    def get_device(self, path: str):
        """
        Returns a device identified by a given id.

        The available devices are determined by the initial session settings.

        The handle to device is invalid after the session is closed
        (i.e. the session object is disposed).

        :param path: a path to the device
        :return: a handle to device
        """
        # TODO use only device available in arrus core
        # First, try getting initially available devices.
        if path in self._py_devices:
            return self._py_devices[path]
        # Then try finding a device using arrus core implementation.

        device_handle = self._session_handle.getDevice(path)

        # Cast device to its type class.
        device_id = device_handle.getDeviceId()
        device_type = device_id.getDeviceType()
        specific_device_cast = {
            # Currently only us4r is supported.
            arrus.core.DeviceType_Us4R:
                lambda handle: arrus.devices.us4r.Us4R(
                    arrus.core.castToUs4r(handle),
                    self)

        }.get(device_type, None)
        if specific_device_cast is None:
            raise arrus.exceptions.DeviceNotFoundError(path)
        specific_device = specific_device_cast(device_handle)
        # TODO(pjarosik) key should be an id, not the whole path
        self._py_devices[path] = specific_device
        return specific_device

    def get_session_context(self):
        return self._context

    def set_current_medium(self, medium: arrus.medium.Medium):
        # TODO mutex, forbid when context is frozen (e.g. when us4r is running)
        raise RuntimeError("NYI")

    def _create_py_devices(self):
        devices = {}
        # Create CPU and GPU devices
        devices["/CPU:0"] = arrus.devices.cpu.CPU(0)
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is not None:
            import cupy
            cupy.cuda.device.Device(0).use()
            devices["/GPU:0"] = arrus.devices.gpu.GPU(0)
        return devices


