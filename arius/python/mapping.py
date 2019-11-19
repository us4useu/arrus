import itertools


def get_esaote_tx_channel_mapping():
    block_size = 32
    card0_mapping = (
        _get_esaote_block_mapping(i, block_size)
        for i in range(0, 128, block_size)
    )
    card0_mapping = list(itertools.chain.from_iterable(card0_mapping))
    card1_mapping = list(range(0, 128))
    return {
        0: card0_mapping,
        1: card1_mapping
    }


def get_esaote_rx_channel_mapping():
    card0_mapping = _get_esaote_block_mapping(0, 32)
    card1_mapping = list(range(0, 32))
    return {
        0: card0_mapping,
        1: card1_mapping
    }


def _get_esaote_block_mapping(origin, block_size):
    block = list(reversed(range(origin, origin+block_size)))
    block[block_size//2-1] = origin+block_size//2-1
    block[block_size//2] = origin+block_size//2
    return block

