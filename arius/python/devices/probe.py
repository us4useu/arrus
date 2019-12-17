import numpy as np
import arius.python.devices.device as _device
import arius.python.devices.arius as _arius
import arius.python.utils as _utils
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

class Subaperture:
    def __init__(self, origin: int, size: int):
        self.origin = origin
        self.size = size

    def __eq__(self, other):
        if not isinstance(other, Subaperture):
            return NotImplementedError

        return self.origin == other.origin and self.size == other.size

    def __str__(self):
        return "Subaperture(origin=%d, size=%d)" % (self.origin, self.size)

    def __repr__(self):
        return self.__str__()

class ProbeHardwareSubaperture(Subaperture):
    def __init__(self, card: _arius.AriusCard, origin: int, size: int):
        super().__init__(origin, size)
        self.card = card


class Probe(_device.Device):
    _DEVICE_NAME = "Probe"

    @staticmethod
    def get_probe_id(index):
        return _device.Device.get_device_id(Probe._DEVICE_NAME, index)

    def __init__(self,
                 index: int,
                 model_name: str,
                 hw_subapertures,
                 pitch,
                 master_card: _arius.AriusCard
    ):
        super().__init__(Probe._DEVICE_NAME, index)
        self.model_name = model_name
        self.hw_subapertures = hw_subapertures
        self.master_card = master_card
        self.n_channels = sum(s.size for s in self.hw_subapertures)
        self.dtype = np.dtype(np.int16)
        self.pitch = pitch

    def start_if_necessary(self):
        for subaperture in self.hw_subapertures:
            subaperture.card.start_if_necessary()

    def get_tx_n_channels(self):
        return self.n_channels

    def get_rx_n_channels(self):
        return self.n_channels

    def transmit_and_record(
            self,
            carrier_frequency: float,
            beam=None,
            tx_aperture: Subaperture=None,
            tx_delays=None,
            n_tx_periods: int = 1,
            n_samples: int=4096,
            rx_time: float=80e-6
    ):
        _utils.assert_true(
            (beam is not None) ^ (tx_aperture is not None and tx_delays is not None),
            "Exactly one of the following parameters should be provided: "
            "beam, (tx aperture, tx delays)"
        )
        if beam is not None:
            beam.set_pitch(self.pitch)
            beam.set_aperture_size(self.get_tx_n_channels())
            tx_aperture, tx_delays = beam.build()
            tx_delays = tx_delays.tolist()
        # Validate input.

        _utils._assert_equal(
            len(tx_delays), tx_aperture.size,
            desc="Array of TX delays should contain %d numbers (probe aperture size)" % tx_aperture.size
        )
        _utils.assert_true(
            carrier_frequency > 0,
            desc="Carrier frequency should be greater than zero."
        )
        _utils.assert_true(
            n_samples > 0,
            desc="Number of samples should be greater than zero."
        )
        _utils.assert_true(
            n_samples % 4096 == 0,
            desc="Number of samples should be a multiple of 4096."
        )
        # Probe's Tx aperture settings.
        tx_start = tx_aperture.origin
        tx_end = tx_aperture.origin + tx_aperture.size
        # We set aperture [tx_start, tx_end)

        current_origin = 0
        delays_origin = 0
        self.log(
            DEBUG,
            "Setting TX aperture: origin=%d, size=%d" % (tx_aperture.origin, tx_aperture.size)
        )
        self.start_if_necessary()
        for s in self.hw_subapertures:
            # Set all TX parameters here (if necessary).
            card, origin, size = s.card, s.origin, s.size

            current_start = current_origin
            current_end = current_origin+size
            if tx_start < current_end and tx_end > current_start:
                # Current boundaries of the PROBE subaperture.
                aperture_start = max(current_start, tx_start)
                aperture_end = min(current_end, tx_end)
                # Origin relative to the start of the CARD's aperture.
                delays_subarray_size = aperture_end-aperture_start

                # Set delays
                # TODO(pjarosik) is it necessary to apply 0.0 to inactive channels?
                # It would be better to keep old values at inactive positions
                card_delays = tx_delays[delays_origin:(delays_origin+delays_subarray_size)]
                card_delays = [0.0]*(origin+aperture_start-current_origin) + card_delays \
                            + [0.0]*(card.get_n_tx_channels() - ((aperture_end-current_origin)+origin))
                card.set_tx_delays(card_delays)

                # Other TX/RX parameters.
                card.set_tx_frequency(carrier_frequency)
                card.set_tx_periods(n_tx_periods)
                card.set_tx_aperture(origin=0, size=0)
                card.set_rx_time(rx_time)

                # Set the final TX Aperture
                card.set_tx_aperture(
                    origin+(aperture_start-current_origin),
                    aperture_end-aperture_start)
                delays_origin += delays_subarray_size
                # TODO(pjarosik) update TX/RX parameters only when it is necessary.
            current_origin += size

        # Rx parameters and an output buffer.
        output = np.zeros((n_samples, self.n_channels), dtype=self.dtype)
        # Cards, which will be used on the RX step.
        # Currently all cards are used to acquire RF data.
        rx_cards = [s.card for s in self.hw_subapertures]
        card_host_buffers = {}
        for rx_card in rx_cards:
            card_host_buffers[rx_card.get_id()] = _utils.create_aligned_array(
                (n_samples, rx_card.get_n_rx_channels()),
                dtype=self.dtype,
                alignment=4096
            )

        subapertures_to_process = [
            (s.card, Subaperture(s.origin, s.size))
            for s in self.hw_subapertures
        ]
        buffer_device_addr = 0
        tx_nr = 0
        while subapertures_to_process:
            self.log(DEBUG, "Performing transmission nr %d." % tx_nr)
            next_subapertures = []
            # Initiate SGDMA.
            for i, (card, s) in enumerate(subapertures_to_process):
                #TODO(pjarosik) below won't work properly, when s.size < card.n_rx_channels
                card.set_rx_aperture(s.origin, card.get_n_rx_channels())

                card.schedule_receive(buffer_device_addr, n_samples)
                s.origin += card.get_n_rx_channels()
                if s.origin < s.size:
                    next_subapertures.append((card, s))

            self.master_card.sw_trigger()
            # Wait until the data is received.
            for card, _ in subapertures_to_process:
                card.wait_until_sgdma_finished()

            # Initiate RX transfer from device to host.
            channel_offset = 0
            for card, s in subapertures_to_process:
                buffer = card_host_buffers[card.get_id()]
                card.transfer_rx_buffer_to_host(
                    dst_array=buffer,
                    src_addr=buffer_device_addr
                )
                current_channel = channel_offset+tx_nr*card.get_n_rx_channels()
                output[:,current_channel:(current_channel+card.get_n_rx_channels())] = buffer
                channel_offset += s.size
            tx_nr += 1
            # Remove completed subapertures.
            subapertures_to_process = next_subapertures

        return output

    def set_tx_delay(self, channel_nr:int, delay:float):
        _utils.assert_true(
            self.get_tx_n_channels() > channel_nr,
            desc="Maximum channel number is %d" % (self.get_tx_n_channels()-1)
        )
        _utils.assert_true(
            delay >= 0.0,
            desc="Delay should be non-negative (is %d)" % delay
        )
        # TODO(pjarosik) optimize below
        current_origin = 0
        for subaperture in self.hw_subapertures:
            current_end = current_origin+subaperture.size
            if current_origin <= channel_nr < current_end:
                return subaperture.card.set_tx_delay(
                    channel=channel_nr-current_origin,
                    delay=float(delay))
            current_origin += subaperture.size
        raise ValueError("Channel not found: %d" % channel_nr)

    def set_tx_aperture(self, tx_aperture):
        _utils.assert_true(
            self.get_tx_n_channels() >= tx_aperture.origin+tx_aperture.size,
            desc="Aperture cannot exceed number of TX channels (%d)" % self.get_tx_n_channels()
        )
        self.log(DEBUG,
            "Setting Probe TX aperture: origin=%d, size=%d" % (tx_aperture.origin, tx_aperture.size)
        )
        tx_start = tx_aperture.origin
        tx_end = tx_aperture.origin + tx_aperture.size
        current_origin = 0
        for s in self.hw_subapertures:
            # Set all TX parameters here (if necessary).
            card, origin, size = s.card, s.origin, s.size
            current_start = current_origin
            current_end = current_origin+size
            if tx_start < current_end and tx_end > current_start:
                # Current boundaries of the PROBE subaperture.
                aperture_start = max(current_start, tx_start)
                aperture_end = min(current_end, tx_end)
                card.set_tx_aperture(
                    origin+(aperture_start-current_origin),
                    aperture_end-aperture_start)
            current_origin += size

    def set_tx_frequency(self, frequency):
        _utils.assert_true(
            frequency > 0,
            desc="Carrier frequency should be greater than zero."
        )
        for card in self._get_cards():
            card.set_tx_frequency(frequency)

    def set_tx_periods(self, n_tx_periods):
        for card in self._get_cards():
            card.set_tx_periods(n_tx_periods)

    def set_rx_time(self, rx_time):
        for card in self._get_cards():
            card.set_rx_time(rx_time)

    def set_pga_gain(self, gain):
        for card in self._get_cards():
            card.set_pga_gain(gain)

    def set_lpf_cutoff(self, cutoff):
        for card in self._get_cards():
            card.set_lpf_cutoff(cutoff)

    def set_active_termination(self, active_termination):
        """
        Sets active termination for all cards handling this probe.
        When active termination is None, the property is disabled.

        :param active_termination: active termination, can be None
        """
        for card in self._get_cards():
            card.set_active_termination(active_termination)

    def set_lna_gain(self, gain):
        for card in self._get_cards():
            card.set_lna_gain(gain)

    def set_dtgc(self, attenuation):
        """
        Sets DTGC for all cards handling this probe.
        When attenuation is None, this property is set to disabled.

        :param attenuation: attenuation, can be None
        """
        for card in self._get_cards():
            card.set_dtgc(attenuation)

    def disable_test_patterns(self):
        for card in self._get_cards():
            card.disable_test_patterns()

    def enable_test_patterns(self):
        for card in self._get_cards():
            card.enable_test_patterns()

    def sync_test_patterns(self):
        for card in self._get_cards():
            card.sync_test_patterns()

    def _get_cards(self):
        card_ids = set()
        cards = []
        for hw_subaperture in self.hw_subapertures:
            card = hw_subaperture.card
            if not card.get_id() in card_ids:
                cards.append(card)
                card_ids.add(card.get_id())
        return cards
