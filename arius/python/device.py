""" ARIUS Devices. """
import numpy as np

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

    def __init__(self, index: int, model_name: str, dependencies: list):
        super().__init__(Probe._DEVICE_NAME, index)
        self.model_name = model_name
        self.dependencies = dependencies

    def transmit_and_record(self):
        output = np.zeros(n_channels, n_samples)
        for i in transmits:
            for card in cards:
                output = card.transmit_and_record()
                output[i:(i+32), :] = output
        return output



class AriusCard(Device):
    _DEVICE_NAME = "Arius"

    @staticmethod
    def get_card_id(index):
        return Device.get_device_id(AriusCard._DEVICE_NAME, index)

    def __init__(self, index: int, device_handle):
        super().__init__(AriusCard._DEVICE_NAME, index)
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










