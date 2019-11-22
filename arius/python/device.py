""" ARIUS Devices. """
import numpy as np
import arius.python.iarius as _iarius
from typing import List


class Device:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    @staticmethod
    def get_device_id(name, index):
        return "%s:%d" % (name, index)

    def get_id(self):
        return Device.get_device_id(self.name, self.index)


class Probe(Device):
    _DEVICE_NAME = "Probe"

    @staticmethod
    def get_probe_id(index):
        return Device.get_device_id(Probe._DEVICE_NAME, index)

    def __init__(self,
                 index: int,
                 model_name: str,
                 hw_subapertures: List[ProbeHardwareSubaperture]):

        super().__init__(Probe._DEVICE_NAME, index)
        self.model_name = model_name
        self.hw_subapertures = hw_subapertures
        self.n_channels = sum(s.size for s in self.hw_subapertures)
        for subaperture in self.hw_subapertures:
            subaperture.card.start()

    def get_tx_n_channels(self):
        return self.n_channels

    def get_rx_n_channels(self):
        return self.n_channels

    def transmit_and_record(self, tx_aperture, n_samples):
        # tx_delay
        # tx_frequency
        # tx_aperture: origin, size

        output = np.zeros(self.n_channels, n_samples)



        for i in transmits:
            for card in cards:
                output = card.transmit_and_record()
                output[i:(i+32), :] = output
        return output


class Subaperture:
    def __init__(self, origin: int, size: int):
        self.origin = origin
        self.size = size


class ProbeHardwareSubaperture(Subaperture):
    def __init__(self, card: AriusCard, origin: int, size: int):
        super().__init__(origin, size)
        self.card = card

def assert_card_is_poweredup(f):
    def wrapper(*args):
        arius_card = args[0]
        if arius_card.is_powered_down():
            raise RuntimeError("Card is powered down. Start the card first.")
        return f(*args)
    return wrapper


class AriusCard(Device):
    _DEVICE_NAME = "Arius"

    @staticmethod
    def get_card_id(index):
        return Device.get_device_id(AriusCard._DEVICE_NAME, index)

    def __init__(self, index: int, card_handle: _iarius.Arius):
        """
        :param index: an index of card
        :param card_handle: a handle to the Arius C++ class
        """
        super().__init__(AriusCard._DEVICE_NAME, index)
        self.card_handle = card_handle

    def start(self):
        """
        Starts the card if is powered down.
        """
        if self.is_powered_down():
            self.card_handle.Powerup()
            self.card_handle.InitializeClocks()
            self.card_handle.InitializeRX()
            self.card_handle.InitializeTX()

    @assert_card_is_poweredup
    def set_tx_channel_mapping(self, tx_channel_mapping: List[int]):
        """
        Sets card's TX channel mapping.

        :param tx_channel_mapping: a list, where
        list[interface channel] = arius card channel
        """
        for dst, src in enumerate(tx_channel_mapping):
            self.card_handle.SetTxChannelMapping(
                srcChannel=src,
                dstChannel=dst
            )

    @assert_card_is_poweredup
    def set_rx_channel_mapping(self, rx_channel_mapping: List[int]):
        """
        Sets card's RX channel mapping.

        :param rx_channel_mapping: a list, where
        list[interface channel] = arius card channel
        """
        for dst, src in enumerate(rx_channel_mapping):
            self.card_handle.SetRxChannelMapping(
                srcChannel=src,
                dstChannel=dst
            )

    @assert_card_is_poweredup
    def set_tx_aperture(self, origin: int, size: int):
        """
        Sets TX aperture.

        :param origin: an origin channel of the aperture
        :param size: a length of the aperture
        """
        self.card_handle.SetTxAperture(origin=origin, size=size)

    @assert_card_is_poweredup
    def set_rx_aperture(self, origin: int, size: int):
        """
        Sets TX aperture.

        :param origin: an origin channel of the aperture
        :param size: a length of the aperture
        """
        self.card_handle.SetRxAperture(origin=origin, size=size)

    @assert_card_is_poweredup
    def set_tx_delay(self, channel: int, delay: float):
        pass

    @assert_card_is_poweredup
    def set_pga_gain(self, gain):
        pass

    def set_rx_time(self, time: float):
        pass

    def schedule_receive(self):
        pass

    def sw_trigger(self):
        pass

    def transfer_rx_buffer_to_host(self):
        pass

    @assert_card_is_poweredup
    def set_tx_frequency(self, frequency: float):
        pass

    @assert_card_is_poweredup
    def set_tx_periods(self, n_periods: int):
        pass

    def is_powered_down(self):
        return self.card_handle.IsPowereddown()

