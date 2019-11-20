import ctypes
import os
import yaml

import arius.python.device as _device
import arius.python.interface as _interface
import arius.python.utils as _utils


_ARIUS_PATH_ENV = "ARIUS_PATH"


class InteractiveSession:
    def __init__(self):
        self._dll = self._load_arius_library("Arius.dll")
        self._devices = self._load_devices("default.yaml")

    def get_device(self, name: str):
        dev_path = name.split("/")[1:]
        if len(dev_path) != 1:
            raise ValueError(
              "Invalid path, currently top-level devices can be accessed only.")
        dev_id = dev_path[0]
        return self._devices[dev_id]

    def _load_devices(self, cfg_file: str, dll):
        """
        Reads configuration from given file and returns a map of top-level
        devices.

        Currently only probes (and required cards) are loaded.

        :param cfg_file: name of the configuration file to read, relative to
                         ARIUS_PATH
        :return: a map: device id -> Device
        """
        result = {}
        path = os.path.join(os.environ[_ARIUS_PATH_ENV], cfg_file)
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        for i, probe_def in enumerate(cfg):
            name, definition = next(iter(probe_def.items()))
            interface_name = definition['interface']
            apertures = definition['aperture']
            interface = _interface.get_interface(interface_name)
            order = interface.get_card_order()
            tx_mappings = interface.get_tx_channel_mapping()
            rx_mappings = interface.get_rx_channel_mapping()


            for i, aperture, tx_m, rx_m in zip(order, apertures,
                                               tx_mappings, rx_mappings):

                # TODO(pjarosik) is it necessary to store card id in cfg?
                _utils.assert_true(
                    i == aperture["card"],
                    "Card mapping order corresponds to the order defined in cfg."
                )
                card_ptr = self._dll.GetArius(i)
                # TODO(pjarosik) use swig


            probe_dependencies = []

            probe = _device.Probe(
                index=i,
                model_name=name,
                dependencies=probe_dependencies
            )
            result[probe.get_id()] = probe

    @staticmethod
    def _load_arius_library(name: str):
        path = os.environ[_ARIUS_PATH_ENV]
        path = os.path.join(path, name)
        return ctypes.cdll.LoadLibrary(path)








