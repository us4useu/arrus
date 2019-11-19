""" ARIUS Devices. """

class Device:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    def get_id(self):
        return "%s:%d" % (self.name, self.index)


class Probe(Device):
    _DEVICE_NAME = "Probe"

    def __init__(self, index: int, host_cards: list):
        super().__init__("Probe", index)
        self.host_cards = host_cards

    def transmit_and_record(self, tx_aperture, tx_delay, tx_frequency, rx_time, pga_gain):
        pass


class AriusCard(Device):
    _DEVICE_NAME = "Arius"

    def __init__(self, index: int, device_handle):
        super().__init__("Arius", index)
        self.device_handle = device_handle

    def set_pga_gain(self, gain):
        pass

    def set_lna_gain(self):
        pass

    def set_lpf_cutoff(self):
        pass

    def set_active_termination(self):
        pass

    def set_dtgc(self):
        pass

    def set_rx_channel_mapping(self):
        pass

    def set_tx_channel_mapping(self):
        pass

    def set_tx_delay(self, channel: int, delay: float):
        pass

    def set_tx_frequency(self, frequency: float):
        pass

    def set_tx_periods(self, n_periods: int):
        pass

    def set_tx_aperture(self, origin: int, size: int):
        pass

    def set_rx_aperture(self, origin: int, size: int):
        pass

    def set_rx_time(self, time: float):
        pass

    def schedule_receive(self):
        # TODO(pjarosik) should be a method of the SGDMA device
        pass

    def sw_trigger(self):
        pass

    def transfer_rx_buffer_to_host(self):
        # TODO(pjarosik) should be a method of the PCIDMA device
        pass










