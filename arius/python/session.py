import ctypes
import logging
import os
from logging import DEBUG, INFO

import yaml

_logger = logging.getLogger(__name__)

import arius.python.devices.probe as _probe
import arius.python.devices.arius as _arius
import arius.python.interface as _interface
import arius.python.utils as _utils
import arius.python.devices.iarius as _iarius



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

    def get_devices(self):
        return self._devices

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
        path = os.path.join(os.environ.get(_ARIUS_PATH_ENV, ""), cfg_file)
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # --- Cards
        n_arius_cards = cfg["nAriusCards"]
        arius_handles = (_iarius.GetArius(i) for i in range(n_arius_cards))
        arius_handles = sorted(arius_handles, key=lambda a: a.GetID())
        arius_cards = [_arius.AriusCard(i, h) for i, h in enumerate(arius_handles)]
        _logger.log(INFO, "Discovered cards: %s" % str(arius_cards))
        for card in arius_cards:
            result[card.get_id()] = card
        master_cards = []
        # --- Probes
        probes = cfg['probes']
        for i, probe_def in enumerate(probes):
            model_name, definition = next(iter(probe_def.items()))
            interface_name = definition['interface']
            apertures = definition['aperture']
            pitch = definition['pitch']
            interface = _interface.get_interface(interface_name)
            order = interface.get_card_order()
            tx_mappings = interface.get_tx_channel_mappings()
            rx_mappings = interface.get_rx_channel_mappings()

            hw_subapertures = []
            master_card = None
            for card_nr, aperture, tx_m, rx_m in zip(order, apertures,
                                                     tx_mappings, rx_mappings):
                _utils.assert_true(
                    card_nr == aperture["card"],
                    "Card mapping order corresponds to the order defined in cfg."
                )

                arius_card = arius_cards[card_nr]
                aperture_origin = aperture["origin"]
                aperture_size = aperture["size"]
                arius_card.store_mappings(tx_m, rx_m)
                # TODO(pjarosik) enable wider range of apertures
                # for example, when subaperture size is smaller
                # than number of card rx channels.
                _utils.assert_true(
                    (aperture_size % arius_card.get_n_rx_channels()) == 0,
                    "Subaperture length should be divisible by %d"
                    " (number of rx channels of device %s)" % (
                        arius_card.get_n_rx_channels(),
                        arius_card.get_id()
                    )
                )
                hw_subapertures.append(
                    _probe.ProbeHardwareSubaperture(
                        arius_card,
                        aperture_origin,
                        aperture_size
                ))
                if aperture.get('master', None):
                    _utils.assert_true(
                        master_card is None,
                        "There should be exactly one master card"
                    )
                    master_card = arius_card
                    master_cards.append(master_card)
            probe = _probe.Probe(
                index=i,
                model_name=model_name,
                hw_subapertures=hw_subapertures,
                pitch=pitch,
                master_card=master_card
            )
            result[probe.get_id()] = probe
            _logger.log(INFO, "Configured %s" % str(probe))
            for hw_subaperture in hw_subapertures:
                card = hw_subaperture.card
                origin = hw_subaperture.origin
                size = hw_subaperture.size
                _logger.log(
                    DEBUG,
                    "---- %s uses %s, origin=%d, size=%d" %
                    (probe, card, origin, size)
                )
        is_hv256 = cfg.get("HV256", None)
        if is_hv256:
            ## -- DBAR and HV256
            _utils.assert_true(
                len(master_cards) == 1,
                "There should be exactly one master card"
            )

            # Intentionally loading modules only when the HV256 is used.
            import arius.python.devices.idbarlite as _dbarlite
            import arius.python.devices.ihv256 as _ihv256
            import arius.python.devices.hv256 as _hv256
            system_master_card = master_cards[0]
            dbar = _dbarlite.IDBARLite(system_master_card.card_handle)
            hv_handle = _ihv256.IHV256(dbar.GetI2CHV())
            hv = _hv256.HV256(hv_handle)
            result[hv.get_id()] = hv
        return result

    @staticmethod
    def _load_arius_library(name: str):
        path = os.environ[_ARIUS_PATH_ENV]
        path = os.path.join(path, name)
        return ctypes.cdll.LoadLibrary(path)


