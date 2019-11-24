""" ARIUS Devices. """
import numpy as np
import time
from typing import List
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

_logger = logging.getLogger(__name__)

import arius.python.iarius as _iarius
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
            self.log(
                INFO,
                "Was powered down, initializing it and powering up...")
            self.card_handle.Powerup()
            self.card_handle.InitializeClocks()
            self.card_handle.InitializeRX()
            self.card_handle.InitializeTX()
            self.log("... successfully powered up.")

    def get_n_rx_channels(self):
        return self.card_handle.GetNRxChannels()

    def get_n_tx_channels(self):
        return self.card_handle.GetNTxChannels()


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
    def set_tx_aperture(self, origin: int, size: int, delays):
        """
        Sets TX aperture.

        :param origin: an origin channel of the aperture
        :param size: a length of the aperture
        """
        self.log(
            DEBUG,
            "Setting TX aperture: origin=%d, size=%d, delays: %s" % (
            origin, size, delays)
        )

        _utils._assert_equal(
            len(delays), size,
            desc="Array of TX delays should contain %d numbers (card aperture size)" % size
        )

        self.card_handle.SetTxAperture(origin=origin, size=size)
        i = origin
        for delay in delays:
            self.card_handle.SetTxDelay(i, delay)
            i += 1

    @assert_card_is_powered_up
    def set_tx_frequency(self, frequency: float):
        self.card_handle.SetTxFreqency(frequency)

    @assert_card_is_powered_up
    def set_tx_periods(self, n_periods: int):
        self.card_handle.SetTxPeriods(n_periods)

    @assert_card_is_powered_up
    def sw_trigger(self, timeout=1):
        self.card_handle.SWTrigger()
        self.log(DEBUG, "Triggered TX/RX scheme.")

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
    def schedule_receive(self, address, length):
        self.log(
            DEBUG,
            "Scheduling data receive at address=0x%02X, length=%d" % (address, length)
        )
        self.card_handle.ScheduleReceive(address, length)

    @assert_card_is_powered_up
    def set_pga_gain(self, gain):
        raise NotImplementedError()

    def set_rx_time(self, time: float):
        self.card_handle.SetRxTime(time)

    @assert_card_is_powered_up
    def transfer_rx_buffer_to_host(self, dst_array, src_addr):
        # TODO(pjarosik) make this method return dst_array
        # instead of passing the result buffer as a method parameter
        dst_addr = dst_array.ctypes.data
        length = dst_array.nbytes
        self.log(
            DEBUG,
            "Transferring %d bytes from RX buffer at 0x%02X to host memory" % (src_addr, length)
        )
        self.card_handle.TransferRXBufferToHost(
            dstAddress=dst_addr,
            length=length,
            srcAddress=src_addr
        )

    def is_powered_down(self):
        return self.card_handle.IsPowereddown()


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
        self.start()

    def start(self):
        for subaperture in self.hw_subapertures:
            subaperture.card.start()

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
                card.set_tx_aperture(
                    origin+(aperture_start-current_origin),
                    aperture_end-aperture_start,
                    delays=tx_delays[delays_origin:(delays_origin+delays_subarray_size)])
                delays_origin += delays_subarray_size
                # TODO(pjarosik) update TX/RX parameters only when it is necessary.
                # Other TX parameters.
                card.set_tx_frequency(carrier_frequency)
                card.set_tx_periods(n_tx_periods)
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
            subapertures_to_remove = set()
            # Initiate SGDMA.
            for i, (card, s) in enumerate(subapertures_to_process):
                #TODO(pjarosik) below won't work properly, when s.size < card.n_rx_channels
                card.set_rx_aperture(s.origin, card.get_n_rx_channels())

                nbytes = card.get_n_rx_channels() \
                         * self.dtype.itemsize \
                         * n_samples

                card.schedule_receive(buffer_device_addr, nbytes)
                s.origin += card.get_n_rx_channels()
                if s.origin >= s.size:
                    subapertures_to_remove.add(i)
            # Trigger master card.
            self.master_card.sw_trigger()
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
            for to_remove in subapertures_to_remove:
                subapertures_to_process.pop(to_remove)
        return output


