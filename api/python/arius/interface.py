import itertools
from typing import Set


class UltrasoundInterface:
    """
    Defines an interface provided by (possibly) multiple Arius modules.
    """
    # TODO(pjarosik) inverse mapping
    def get_card_order(self):
        # TODO(pjarosik) deprecated
        raise NotImplementedError

    def get_tx_channel_mapping(self, module_idx):
        """
        Returns TX channel mapping for a module with given index.

        :param module_idx: module index
        :return: TX channel mapping: a list, where list[probe's channel] = arius module channel.
        """
        raise NotImplementedError

    def get_rx_channel_mapping(self, module_idx):
        """
        Returns RX channel mapping for a module with given index.

        :param module_idx: module index
        :return: RX channel mapping: a list, where list[probe's channel] = arius module channel.
        """
        raise NotImplementedError


class EsaoteInterface(UltrasoundInterface):

    def __init__(self):
        self.cards = (0, 1)
        # A list of lists;
        # for each list, list[interface channel] = arius card channel.
        self.tx_card_channels = self._compute_tx_channel_mapping()
        self.rx_card_channels = self._compute_rx_channel_mapping()

    def get_card_order(self):
        return self.cards

    def get_tx_channel_mappings(self):
        return self.tx_card_channels

    def get_rx_channel_mappings(self):
        return self.rx_card_channels

    def get_tx_channel_mapping(self, module_idx):
        return self.tx_card_channels[module_idx]

    def get_rx_channel_mapping(self, module_idx):
        return self.rx_card_channels[module_idx]

    def _compute_tx_channel_mapping(self):
        block_size = 32
        card0_mapping = (
            self._get_esaote_block_mapping(i, block_size)
            for i in range(0, 128, block_size)
        )
        card0_mapping = list(itertools.chain.from_iterable(card0_mapping))
        card1_mapping = list(range(0, 128))
        return [
            card0_mapping,
            card1_mapping
        ]

    def _compute_rx_channel_mapping(self):
        card0_mapping = self._get_esaote_block_mapping(0, 32)
        card1_mapping = list(range(0, 32))
        return [
            card0_mapping,
            card1_mapping
        ]

    @staticmethod
    def _get_esaote_block_mapping(origin, block_size):
        block = list(reversed(range(origin, origin+block_size)))
        block[block_size//2-1] = origin+block_size//2-1
        block[block_size//2] = origin+block_size//2
        return block


class UltrasonixInterface(UltrasoundInterface):
    def __init__(self):
        self.cards = (0, 1)
        # A list of lists;
        # for each list, list[interface channel] = arius card channel.
        self.tx_card_channels = self._compute_tx_channel_mapping()
        self.rx_card_channels = self._compute_rx_channel_mapping()

    def get_card_order(self):
        return self.cards

    def get_tx_channel_mappings(self):
        return self.tx_card_channels

    def get_rx_channel_mappings(self):
        return self.rx_card_channels

    def get_tx_channel_mapping(self, module_idx):
        return self.tx_card_channels[module_idx]

    def get_rx_channel_mapping(self, module_idx):
        return self.rx_card_channels[module_idx]

    def _compute_tx_channel_mapping(self):
        block_size = 32
        card0_mapping = list(range(0, 128))
        card1_mapping = (
            self._get_ultrasonix_block_mapping(i, block_size)
            for i in range(0, 128, block_size)
        )
        card1_mapping = list(itertools.chain.from_iterable(card1_mapping))
        return [
            card0_mapping,
            card1_mapping
        ]
    def _compute_rx_channel_mapping(self):
        card0_mapping = list(range(0, 32))
        card1_mapping = self._get_ultrasonix_block_mapping(0, 32)
        return [
            card0_mapping,
            card1_mapping
        ]

    @staticmethod
    def _get_ultrasonix_block_mapping(origin, block_size):
        block = list(reversed(range(origin, origin+block_size)))
        return block


_INTERFACES = {
    'esaote': EsaoteInterface(),
    'ultrasonix': UltrasonixInterface()
}

def get_interface(name: str):
    """
    Returns an interface registered in arius-sdk under given name.

    :param name: name to the interface
    :return: an interface object
    """
    return _INTERFACES[name]

