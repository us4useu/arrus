import ctypes
import os
import yaml

import arius.python.device as _device
import arius.python.interface as _interface
import arius.python.utils as _utils
import arius.python.iarius as _iarius


_ARIUS_PATH_ENV = "ARIUS_PATH"


class InteractiveSession:
    def __init__(self):
        self._devices = self._load_devices("default.yaml")

    def get_device(self, id: str):
        """
        Returns device from given path.

        Currently, ONLY TOP-LEVEL DEVICES ARE AVAILABLE.

        :param a path to a device
        :return: a device located in given path.
        """
        dev_path = id.split("/")[1:]
        if len(dev_path) != 1:
            raise ValueError(
              "Invalid path, top-level devices can be accessed only.")
        dev_id = dev_path[0]
        return self._devices[dev_id]

    @staticmethod
    def _load_devices(cfg_file: str):
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

        n_arius_cards = cfg["nAriusCards"]
        arius_handles = (_iarius.Arius(i) for i in range(n_arius_cards))
        arius_handles = sorted(arius_handles, key=lambda a: a.GetId())
        arius_cards = [_device.AriusCard(i, h) for i, h in enumerate(arius_handles)]
        for card in arius_cards:
            result[card.get_id()] = card

        probes = cfg['probes']
        for i, probe_def in enumerate(probes):
            model_name, definition = next(iter(probe_def.items()))
            interface_name = definition['interface']
            apertures = definition['aperture']
            interface = _interface.get_interface(interface_name)
            order = interface.get_card_order()
            tx_mappings = interface.get_tx_channel_mapping()
            rx_mappings = interface.get_rx_channel_mapping()

            hw_subapertures = []
            for card_nr, aperture, tx_m, rx_m in zip(order, apertures,
                                                     tx_mappings, rx_mappings):
                _utils.assert_true(
                    card_nr == aperture["card"],
                    "Card mapping order corresponds to the order defined in cfg."
                )
                arius_card = arius_cards[card_nr]
                arius_card.set_tx_channel_mapping(tx_m)
                arius_card.set_rx_channel_mapping(rx_m)
                hw_subapertures.append(
                    _device.ProbeHardwareSubaperture(
                        arius_card,
                        aperture["origin"],
                        aperture["size"]
                ))
            probe = _device.Probe(
                index=i,
                model_name=model_name,
                hw_subapertures=hw_subapertures
            )
            result[probe.get_id()] = probe
        return result

    @staticmethod
    def _load_arius_library(name: str):
        path = os.environ[_ARIUS_PATH_ENV]
        path = os.path.join(path, name)
        return ctypes.cdll.LoadLibrary(path)








