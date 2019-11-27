""" ARIUS Devices. """
import numpy as np
import time
import ctypes
from typing import List
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

_logger = logging.getLogger(__name__)

import arius.python.iarius as _iarius
import arius.python.hv256 as _hv256
import arius.python.utils  as _utils


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


class Device:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    @staticmethod
    def get_device_id(name, index):
        if index is None:
            return name
        else:
            return "%s:%d" % (name, index)

    def get_id(self):
        return Device.get_device_id(self.name, self.index)

    def log(self, level, msg):
        _logger.log(level, "%s: %s" % (self.get_id(), msg))

    def __str__(self):
        return self.get_id()

    def __repr__(self):
        return self.__str__()


def assert_card_is_powered_up(f):
    def wrapper(*args, **kwargs):
        arius_card = args[0]
        if arius_card.is_powered_down():
            raise RuntimeError("Card is powered down. Start the card first.")
        return f(*args, **kwargs)
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

    def start_if_necessary(self):
        """
        Starts the card if is powered down.
        """
        if self.is_powered_down():
            self.log(
                INFO,
                "Was powered down, initializing it and powering up...")
            self.card_handle.Powerup()
            self.card_handle.InitializeClocks()
            self.card_handle.InitializeDDR4()
            self.card_handle.InitializeRX()
            self.card_handle.InitializeTX()
            self.set_tx_channel_mapping(self.tx_channel_mapping)
            self.set_rx_channel_mapping(self.rx_channel_mapping)
            self.log(INFO, "... successfully powered up.")

    def get_n_rx_channels(self):
        return self.card_handle.GetNRxChannels()

    def get_n_tx_channels(self):
        return self.card_handle.GetNTxChannels()

    def store_mappings(self, tx_m, rx_m):
        self.tx_channel_mapping = tx_m
        self.rx_channel_mapping = rx_m


    @assert_card_is_powered_up
    def set_tx_channel_mapping(self, tx_channel_mapping: List[int]):
        """
        Sets card's TX channel mapping.

        :param tx_channel_mapping: a list, where
        list[interface channel] = arius card channel
        """
        self.log(DEBUG, "Setting TX channel mapping: %s" % str(tx_channel_mapping))
        for dst, src in enumerate(tx_channel_mapping):
            self.card_handle.SetTxChannelMapping(
                srcChannel=src,
                dstChannel=dst
            )

    @assert_card_is_powered_up
    def set_rx_channel_mapping(self, rx_channel_mapping: List[int]):
        """
        Sets card's RX channel mapping.

        :param rx_channel_mapping: a list, where
        list[interface channel] = arius card channel
        """
        self.log(DEBUG, "Setting RX channel mapping: %s" % str(rx_channel_mapping))
        for dst, src in enumerate(rx_channel_mapping):
            self.card_handle.SetRxChannelMapping(
                srcChannel=src,
                dstChannel=dst
            )

    @assert_card_is_powered_up
    def set_tx_aperture(self, origin: int, size: int):
        """
        Sets TX aperture.

        :param origin: an origin channel of the aperture
        :param size: a length of the aperture
        """
        self.log(
            DEBUG,
            "Setting TX aperture: origin=%d, size=%d" % (origin, size)
        )

        self.card_handle.SetTxAperture(origin=origin, size=size)

    @assert_card_is_powered_up
    def set_tx_delays(self, delays):
        _utils._assert_equal(
            len(delays), self.get_n_tx_channels(),
            desc="Array of TX delays should contain %d numbers (card number of TX channels)"
                 % self.get_n_tx_channels()
        )
        self.log(DEBUG, "Setting TX delays: %s" % (delays))
        for i, delay in enumerate(delays):
            self.card_handle.SetTxDelay(i, delay)

    @assert_card_is_powered_up
    def set_tx_frequency(self, frequency: float):
        self.log(
            DEBUG,
            "Setting TX frequency: %f" % frequency
        )
        self.card_handle.SetTxFreqency(frequency)

    @assert_card_is_powered_up
    def set_tx_periods(self, n_periods: int):
        self.log(
            DEBUG,
            "Setting number of bursts: %f" % n_periods
        )
        self.card_handle.SetTxPeriods(n_periods)

    @assert_card_is_powered_up
    def sw_trigger(self):
        self.card_handle.SWTrigger()
        self.log(DEBUG, "Triggered single wave.")

    @assert_card_is_powered_up
    def wait_until_sgdma_finished(self, timeout=1):
        start = time.time()
        self.log(DEBUG, "Waiting till all data are received...")
        # TODO(pjarosik) active waiting should be avoided here
        while not self.card_handle.IsReceived():
            if time.time() - start > timeout:
                raise TimeoutError("Timeout while waiting for RX data.")
            time.sleep(0.001)
        self.log(DEBUG, "...done.")

    @assert_card_is_powered_up
    def set_rx_aperture(self, origin: int, size: int):
        """
        Sets TX aperture.

        :param origin: an origin channel of the aperture
        :param size: a length of the aperture
        """
        self.log(
            DEBUG,
            "Setting RX aperture: origin=%d, size=%d" % (origin, size)
        )
        self.card_handle.SetRxAperture(origin=origin, size=size)

    @assert_card_is_powered_up
    def set_rx_time(self, time: float):
        self.log(
            DEBUG,
            "Setting RX time: %f" % time
        )
        self.card_handle.SetRxTime(time)

    @assert_card_is_powered_up
    def schedule_receive(self, address, length):
        self.log(
            DEBUG,
            "Scheduling data receive at address=0x%02X, length=%d" % (address, length)
        )
        self.card_handle.ScheduleReceive(address, length)

    @assert_card_is_powered_up
    def transfer_data_to_rx_buffer(self, address, length):
        self.log(
            DEBUG,
            "Transfer RX data to buffer at address=0x%02X, length=%d" % (address, length)
        )
        self.card_handle.Receive(address, length)

    @assert_card_is_powered_up
    def set_pga_gain(self, gain):
        self.log(
            DEBUG,
            "Setting PGA Gain: %d" % gain
        )
        enum_value = self._convert_to_enum_value(
            enum_name="PGA_GAIN",
            value=gain,
            unit="dB"
        )
        self.card_handle.SetPGAGain(enum_value)

    @assert_card_is_powered_up
    def set_lpf_cutoff(self, cutoff):
        self.log(
            DEBUG,
            "Setting LPF Cutoff: %d" % cutoff
        )
        enum_value = self._convert_to_enum_value(
            enum_name="LPF_PROG",
            value=cutoff,
            unit="MHz"
        )
        self.card_handle.SetLPFCutoff(cutoff)

    @assert_card_is_powered_up
    def set_active_termination(self, active_termination):
        """
        Sets active termination for this card. When active termination is None,
        the property is disabled.

        :param active_termination: active termination, can be None
        """
        self.log(
            DEBUG,
            "Setting active termination: %d" % active_termination
        )
        if active_termination:
            enum_value = self._convert_to_enum_value(
                enum_name="GBL_ACTIVE_TERM",
                value=active_termination,
            )
            self.card_handle.SetActiveTermination(
                endis=_iarius.ACTIVE_TERM_EN_ACTIVE_TERM_EN,
                term=enum_value
            )
        else:
            self.card_handle.SetActiveTermination(
                endis=_iarius.ACTIVE_TERM_EN_ACTIVE_TERM_DIS,
                # TODO(pjarosik) when disabled, what value should be set?
                term=_iarius.GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_50
            )

    @assert_card_is_powered_up
    def set_lna_gain(self, gain):
        self.log(
            DEBUG,
            "Setting LNA Gain: %d" % gain
        )
        enum_value = self._convert_to_enum_value(
            enum_name="LNA_GAIN_GBL",
            value=gain,
            unit="dB"
        )

    @assert_card_is_powered_up
    def set_dtgc(self, attenuation):
        """
        Sets DTGC for this card. When attenuation is None, this property
        is set to disabled.

        :param active_termination: attenuation, can be None
        """
        self.log(
            DEBUG,
            "Setting DTGC: %d" % attenuation
        )
        if attenuation:
            enum_value = self._convert_to_enum_value(
                enum_name="DIG_TGC_ATTENUATION",
                value=attenuation,
                unit="dB"
            )
            self.card_handle.SetDTGC(
                endis=_iarius.EN_DIG_TGC_EN_DIG_TGC_EN,
                att=attenuation
            )
        else:
            self.card_handle.SetDTGC(
                endis=_iarius.EN_DIG_TGC_EN_DIG_TGC_DIS,
                att=_iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_0dB
            )

    @assert_card_is_powered_up
    def enable_test_patterns(self):
        self.log(
            DEBUG,
            "Enabling Test Patterns."
        )
        self.card_handle.EnableTestPatterns()

    @assert_card_is_powered_up
    def disable_test_patterns(self):
        self.log(
            DEBUG,
            "Disabling Test Patterns."
        )
        self.card_handle.DisableTestPatterns()

    @assert_card_is_powered_up
    def sync_test_patterns(self):
        self.log(
            DEBUG,
            "Syncing with test patterns..."
        )
        self.card_handle.SyncTestPatterns()

    @assert_card_is_powered_up
    def transfer_rx_buffer_to_host(self, dst_array, src_addr):
        # TODO(pjarosik) make this method return dst_array
        # instead of passing the result buffer as a method parameter
        dst_addr = dst_array.ctypes.data
        length = dst_array.nbytes
        self.log(
            DEBUG,
            "Transferring %d bytes from RX buffer at 0x%02X to host memory..." % (length, src_addr)
        )
        self.card_handle.TransferRXBufferToHostLocation(
            dstAddress=dst_addr,
            length=length,
            srcAddress=src_addr+0x100000000
        )
        self.log(
            DEBUG,
            "... transferred."
        )

    def is_powered_down(self):
        return self.card_handle.IsPowereddown()

    def _convert_to_enum_value(self, enum_name, value, unit=""):
        _utils.assert_true(
            round(value) == value,
            "Value %s for '%s' should be an integer value." % (value, enum_name)
        )
        const_prefix = enum_name + "_" + enum_name
        const_name = const_prefix + "_" + str(value) + unit
        try:
            return getattr(_iarius, const_name)
        except AttributeError:
            acceptable_values = set()
            for key in dir(_iarius):
                if key.startswith(const_prefix):
                    value_with_unit = key[len(const_prefix)+1:]
                    value = int("".join(list(filter(str.isdigit, value_with_unit))))
                    acceptable_values.add(value)
            if not acceptable_values:
                raise RuntimeError("Invalid enum name: % s" % enum_name)
            else:
                raise ValueError(
                    "Invalid value for %s, should be one of the following: %s" %
                    (enum_name, str(acceptable_values))
                )


class ProbeHardwareSubaperture(Subaperture):
    def __init__(self, card: AriusCard, origin: int, size: int):
        super().__init__(origin, size)
        self.card = card


class Probe(Device):
    _DEVICE_NAME = "Probe"

    @staticmethod
    def get_probe_id(index):
        return Device.get_device_id(Probe._DEVICE_NAME, index)

    def __init__(self,
                 index: int,
                 model_name: str,
                 hw_subapertures,
                 master_card: AriusCard
    ):

        super().__init__(Probe._DEVICE_NAME, index)
        self.model_name = model_name
        self.hw_subapertures = hw_subapertures
        self.master_card = master_card
        self.n_channels = sum(s.size for s in self.hw_subapertures)
        self.dtype = np.dtype(np.int16)

    def start_if_necessary(self):
        for subaperture in self.hw_subapertures:
            subaperture.card.start_if_necessary()

    def get_tx_n_channels(self):
        return self.n_channels

    def get_rx_n_channels(self):
        return self.n_channels

    def transmit_and_record(
            self,
            tx_aperture: Subaperture,
            tx_delays,
            carrier_frequency: float,
            n_tx_periods: int = 1,
            n_samples: int=4096,
            rx_time: float=80e-6
    ):
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
            # Other RX parameters.
            rx_card.set_rx_time(rx_time)

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

                nbytes = card.get_n_rx_channels() \
                         * self.dtype.itemsize \
                         * n_samples

                card.schedule_receive(buffer_device_addr, nbytes)
                s.origin += card.get_n_rx_channels()
                if s.origin < s.size:
                    next_subapertures.append((card, s))
            # Trigger master card.
            for card, _ in subapertures_to_process:
                card.sw_trigger()
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

    def set_pga_gain(self, gain):
        self.start_if_necessary()
        for card in self._get_cards():
            card.set_pga_gain(gain)

    def set_lpf_cutoff(self, cutoff):
        self.start_if_necessary()
        for card in self._get_cards():
            card.set_lpf_cutoff(cutoff)

    def set_active_termination(self, active_termination):
        """
        Sets active termination for all cards handling this probe.
        When active termination is None, the property is disabled.

        :param active_termination: active termination, can be None
        """
        self.start_if_necessary()
        for card in self._get_cards():
            card.set_active_termination(active_termination)

    def set_lna_gain(self, gain):
        self.start_if_necessary()
        for card in self._get_cards():
            card.set_lna_gain(gain)

    def set_dtgc(self, attenuation):
        """
        Sets DTGC for all cards handling this probe.
        When attenuation is None, this property is set to disabled.

        :param attenuation: attenuation, can be None
        """
        self.start_if_necessary()
        for card in self._get_cards():
            card.set_dtgc(attenuation)

    def disable_test_patterns(self):
        self.start_if_necessary()
        for card in self._get_cards():
            card.disable_test_patterns()

    def enable_test_patterns(self):
        self.start_if_necessary()
        for card in self._get_cards():
            card.enable_test_patterns()

    def sync_test_patterns(self):
        self.start_if_necessary()
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


class HV256(Device):
    _DEVICE_NAME = "HV256"

    @staticmethod
    def get_card_id(index):
        return Device.get_device_id(AriusCard._DEVICE_NAME, index)

    def __init__(self, hv256_handle: _hv256.HV256):
        """
        HV 256 Device. Provides means to steer the voltage
        set on the master Arius card.

        :param card_handle: a handle to the HV256 C++ class.
        """
        super().__init__(HV256._DEVICE_NAME, index=None)
        self.hv256_handle = hv256_handle

    def enable_hv(self):
        self.hv256_handle.EnableHV()

    def disable_hv(self):
        self.hv256_handle.DisableHV()

    def set_hv_voltage(self, voltage):
        self.hv256_handle.SetHVVoltage(voltage=voltage)
