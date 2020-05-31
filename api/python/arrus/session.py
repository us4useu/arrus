import logging
import os
from logging import DEBUG, INFO
import abc
import dataclasses
import typing

import yaml

_logger = logging.getLogger(__name__)

import arrus.devices.probe as _probe
import arrus.devices.us4oem as _us4oem
import arrus.devices.device as _device
import arrus.interface as _interface
import arrus.utils as _utils
import arrus.devices.ius4oem as _ius4oem
import arrus.system as _system
import arrus.kernels
import arrus.validation

_ARRUS_PATH_ENV = "ARRUS_PATH"



@dataclasses.dataclass(frozen=True)
class SessionCfg:
    """
    Configuration of the communication session with devices.

    This configuration allows to specify parameters that has to
    applied when user starts a new session.

    :param system: description of a system with which the user wants to
        communicate
    :param devices: configuration of the devices with which the user wants
        communicate
    """
    system: _system.SystemCfg
    devices: typing.Mapping[str, _device.DeviceCfg]

    def __post_init__(self):
        arrus.validation.assert_not_none(self.devices, "devices")
        arrus.validation.assert_not_none(self.devices, "system")


class AbstractSession(abc.ABC):

    def __init__(self, cfg):
        self._devices = self._load_devices(cfg)

    def get_device(self, id: str):
        """
        Returns a device located at given path.

        :param id: a path to a device, for example '/Us4OEM:0'
        :return: a device located in given path.
        """
        dev_path = id.split("/")[1:]
        if len(dev_path) != 1:
            raise ValueError(
              "Invalid path, top-level devices can be accessed only.")
        dev_id = dev_path[0]
        return self._devices[dev_id]

    def get_devices(self):
        """
        Returns a list of all devices available in this session.

        :return: a list of available devices
        """
        return self._devices

    @abc.abstractmethod
    def run(self, operation: arrus.ops.Operation, feed_dict: dict):
        raise ValueError("This type of session cannot run operations.")

    @abc.abstractmethod
    def _load_devices(self, cfg):
        pass


class Session(AbstractSession):
    """
    A communication session with the available devices.
    """

    def __init__(self, cfg: SessionCfg):
        """
        :param cfg: configuration parameters for the new session
        """
        super().__init__(cfg)
        self._async_kernels = {}

    def run(self, operation: arrus.ops.Operation, feed_dict: dict):
        """
        Runs a given operation in the system.

        :param operation: operation to run
        :param feed_dict: values to be set in the place of placeholders.
        """
        _logger.log(DEBUG, f"Session run: {str(operation)}")

        kernel = arrus.kernels.get_kernel(operation, feed_dict)
        device = feed_dict.get("device", None)
        arrus.validation.assert_not_none(device, "device")
        current_async_kernel = self._async_kernels.get(device.get_id(), None)
        if current_async_kernel is not None:
            device_id = device.get_id()
            current_op = current_async_kernel.op
            raise ValueError(f"An operation {current_op} is already running on "
                             f"the device {device_id}. Stop the device first.")

        result = kernel.run()
        if kernel is arrus.kernels.AsyncKernel:
            self._async_kernels[device.get_id()] = kernel
        return result

    def stop_device(self, device: _device.Device):
        """
        Stops executing any operations that run on a specific device.
        """
        current_kernel = self._async_kernels.get(device.get_id(), None)
        if current_kernel is not None:
            op = current_kernel.op
            current_kernel.stop()
            _logger.info(f"Stopped operation {op} running on "
                         f"device {device.get_id()}")

    def stop(self):
        for key, kernel in self._async_kernels:
            _logger.debug(INFO, "Stopping device %s (operation %s)" %
                          (key, kernel.op))
            kernel.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _load_devices(self, cfg: SessionCfg):
        result = {}
        system_cfg = cfg.system

        if not isinstance(system_cfg, _system.CustomUs4RCfg):
            raise ValueError("Only custom us4r configuration is currently "
                             "supported.")

        n_us4oems = system_cfg.n_us4oems
        arrus.validation.assert_not_none(n_us4oems, "number of us4oems")

        us4oem_handles = (_ius4oem.getUs4OEMPtr(i) for i in range(n_us4oems))
        us4oem_handles = sorted(us4oem_handles, key=lambda a: a.GetID())

        for device_id, device_cfg in cfg.devices.items():
            if device_id.strip().upper().startswith("US4OEM"):
                # parse id
                device_type, index = device_id.split(":")
                index = int(index.strip())
                arrus.validation.assert_type(device_cfg, _us4oem.Us4OEMCfg,
                                             f"{device_id} configuration")
                # validate and use configuration to create device
                us4oem = _us4oem.Us4OEM(index,card_handle=us4oem_handles[index],
                                        cfg=device_cfg)
                if us4oem.get_id() in result.keys():
                    raise ValueError(f"Found duplicated configuration for "
                                     f"{us4oem.get_id()}")
                result[us4oem.get_id()] = us4oem
                _logger.log(INFO, f"Discovered module: {str(us4oem)}")
            else:
                raise ValueError(f"This device cannot be configured: "
                                 f"{device_id}")

        if system_cfg.is_hv256:
            module_id = system_cfg.master_us4oem
            master_module = result[_us4oem.Us4OEM.get_card_id(module_id)]

            # Intentionally loading modules only when the HV256 is used.
            import arrus.devices.idbarlite as _dbarlite
            import arrus.devices.ihv256 as _ihv256
            import arrus.devices.hv256 as _hv256

            dbar = _dbarlite.GetDBARLite(
                _ius4oem.castToII2CMaster(master_module.card_handle))
            hv_handle = _ihv256.GetHV256(dbar.GetI2CHV())
            hv = _hv256.HV256(hv_handle)
            result[hv.get_id()] = hv
        return result


class InteractiveSession(AbstractSession):
    """
    **THIS CLASS IS DEPRECATED AND WILL BE REMOVED IN THE NEAR FUTURE. Please
    use :class:`arrus.session.Session`**.

    An user interactive session with available devices.

    If cfg_path is None, session looks for a file ``$ARRUS_PATH/default.yaml``,
    where ARRUS_PATH is an user-defined environment variable.

    :param cfg_path: path to the configuration file, can be None
    """
    def __init__(self, cfg_path: str = None):
        if cfg_path is None:
            cfg_path = os.path.join(os.environ.get(_ARRUS_PATH_ENV, ""), "default.yaml")

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            super().__init__(cfg)

    def run(self, operation: arrus.ops.Operation, feed_dict: dict):
        raise ValueError("This type of session cannot run operations.")

    def _load_devices(self, cfg):
        """
        Reads configuration from given file and returns a map of top-level
        devices.

        Currently Us4OEM modules are supported.

        :param cfg: configuration to read
        :return: a map: device id -> Device
        """
        result = {}
        # --- Cards
        n_us4oems = cfg["nModules"]
        us4oem_handles = (_ius4oem.getUs4OEMPtr(i) for i in range(n_us4oems))
        us4oem_handles = sorted(us4oem_handles, key=lambda a: a.GetID())
        us4oems = [_us4oem.Us4OEM(i, h) for i, h in enumerate(us4oem_handles)]
        _logger.log(INFO, "Discovered modules: %s"%str(us4oems))
        for us4oem in us4oems:
            result[us4oem.get_id()] = us4oem

        is_hv256 = cfg.get("HV256", None)
        if is_hv256:
            module_id = cfg["masterModule"]
            master_module = result[_us4oem.Us4OEM.get_card_id(module_id)]

            # Intentionally loading modules only when the HV256 is used.
            import arrus.devices.idbarlite as _dbarlite
            import arrus.devices.ihv256 as _ihv256
            import arrus.devices.hv256 as _hv256

            dbar = _dbarlite.GetDBARLite(
                _ius4oem.castToII2CMaster(master_module.card_handle))
            hv_handle = _ihv256.GetHV256(dbar.GetI2CHV())
            hv = _hv256.HV256(hv_handle)
            result[hv.get_id()] = hv
        return result
