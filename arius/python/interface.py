import itertools


def get_interface(name: str):
    return _INTERFACES[name]


class UltrasoundInterface:
    """
    Defines an interface provided by multiple Arius cards.
    TODO(pjarosik) inverse mapping?
    """
    def get_card_order(self):
        raise NotImplementedError

    def get_tx_channel_mapping(self, card_idx):
        raise NotImplementedError

    def get_rx_channel_mapping(self, card_idx):
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

    def get_tx_channel_mapping(self, card_idx):
        return self.tx_card_channels[card_idx]

    def get_rx_channel_mapping(self, card_idx):
        return self.rx_card_channels[card_idx]

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


_INTERFACES: {
    'esaote': EsaoteInterface()
}
