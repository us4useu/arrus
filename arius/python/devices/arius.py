from logging import DEBUG, INFO
from typing import List
import numpy as np
from functools import wraps

import arius.python.devices.device as _device
import arius.python.devices.iarius as _iarius
import arius.python.utils as _utils
import math


def assert_card_is_powered_up(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        arius_card = args[0]
        if arius_card.is_powered_down():
            raise RuntimeError("Card is powered down. Start the card first.")
        return f(*args, **kwargs)
    return wrapper


class AriusCard(_device.Device):
    """
    A single Arius module.
    """

    _DEVICE_NAME = "Arius"

    @staticmethod
    def get_card_id(index):
        return _device.Device.get_device_id(AriusCard._DEVICE_NAME, index)

    def __init__(self, index: int, card_handle: _iarius.IArius):
        super().__init__(AriusCard._DEVICE_NAME, index)
        self.card_handle = card_handle
        self.dtype = np.dtype(np.int16)
        self.host_buffer = np.array([])

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
        """
        Returns number of RX channels.

        :return: number of RX channels.
        """
        return 32 # TODO(pjarosik) should be returned by iarius
        # return self.card_handle.GetNRxChannels()

    def get_n_tx_channels(self):
        """
        Returns number of TX channels.

        :return: number of TX channels
        """
        return 128 # TODO(pjarosik) should be returned by iarius
        # return self.card_handle.GetNTxChannels()

    def store_mappings(self, tx_m, rx_m):
        self.tx_channel_mapping = tx_m
        self.rx_channel_mapping = rx_m

    @assert_card_is_powered_up
    def set_tx_channel_mapping(self, tx_channel_mapping: List[int]):
        """
        Sets card's TX channel mapping.

        :param tx_channel_mapping: a list, where list[interface channel] = arius card channel
        """
        _utils.assert_true(
            tx_channel_mapping is not None,
            "TX channel mapping should be not None."
        )
        self.log(DEBUG, "Setting TX channel mapping: %s" % str(tx_channel_mapping))
        self.tx_channel_mapping = tx_channel_mapping
        for dst, src in enumerate(tx_channel_mapping):
            self.card_handle.SetTxChannelMapping(
                srcChannel=src,
                dstChannel=dst
            )

    @assert_card_is_powered_up
    def set_rx_channel_mapping(self, rx_channel_mapping: List[int]):
        """
        Sets card's RX channel mapping.

        :param rx_channel_mapping: a list, where list[interface channel] = arius card channel
        """
        _utils.assert_true(
            rx_channel_mapping is not None,
            "RX channel mapping should be not None."
        )
        self.log(DEBUG, "Setting RX channel mapping: %s" % str(rx_channel_mapping))
        self.rx_channel_mapping = rx_channel_mapping
        for dst, src in enumerate(rx_channel_mapping):
            self.card_handle.SetRxChannelMapping(
                srcChannel=src,
                dstChannel=dst
            )

    @assert_card_is_powered_up
    def set_tx_aperture(self, origin: int, size: int, firing: int=0):
        """
        Sets position of an active TX aperture.

        :param origin: an origin channel of the aperture, **starts from 0**
        :param size: a length of the aperture
        :param firing: a firing, in which the delay should apply, **starts from 0**
        """
        self.log(
            DEBUG,
            "Setting TX aperture: origin=%d, size=%d" % (origin, size)
        )
        self.card_handle.SetTxAperture(origin, size, firing)

    @assert_card_is_powered_up
    def set_tx_delays(self, delays, firing: int=0):
        """
        Sets channel's TX delays.

        :param delays: an array of delays to set [s], number of elements
                       should be equal to the number of module's TX channels
        :param firing: a firing, in which the delay should apply, **starts from 0**
        :return: an array of values, which were set [s]
        """
        _utils._assert_equal(
            len(delays), self.get_n_tx_channels(),
            desc="Array of TX delays should contain %d numbers (card number of TX channels)"
                 % self.get_n_tx_channels()
        )
        self.log(DEBUG, "Setting TX delays: %s" % (delays))
        result_values = []
        for i, delay in enumerate(delays):
            value = self.card_handle.SetTxDelay(i, delay, firing)
            result_values.append(value)
        self.log(DEBUG, "Applied TX delays: %s" % (result_values))
        return result_values

    @assert_card_is_powered_up
    def set_tx_delay(self, channel: int, delay: float, firing: int=0):
        """
        Sets channel's TX delay.

        :param channel: card's channel number
        :param delay: delay to set [s]
        :param firing: a firing, in which the delay should apply, **starts from 0**
        :return: a value, which was set [s]
        """
        _utils.assert_true(
            channel < self.get_n_tx_channels(),
            desc="Channel number cannot exceed number of available TX channels (%d)"
                 % self.get_n_tx_channels()
        )
        self.log(DEBUG, "Setting TX delay: channel=%d, delay=%d" % (channel, delay))
        value = self.card_handle.SetTxDelay(channel, delay, firing)
        return value

    @assert_card_is_powered_up
    def set_tx_frequency(self, frequency: float, firing: int=0):
        """
        Sets TX frequency.

        :param frequency: frequency to set [Hz]
        :param firing: a firing, in which the value should apply, **starts from 0**
        :return: a value, which was set [Hz]
        """
        self.log(
            DEBUG,
            "Setting TX frequency: %f" % frequency
        )
        self.card_handle.SetTxFreqency(frequency, firing)

    @assert_card_is_powered_up
    def set_tx_periods(self, n_periods: int, firing: int=0):
        """
        Sets number of TX periods.

        :param n_periods: number of periods to set
        :param firing: a firing, in which the value should apply, **starts from 0**
        """
        self.log(
            DEBUG,
            "Setting number of bursts: %f" % n_periods
        )
        self.card_handle.SetTxPeriods(n_periods, firing)

    @assert_card_is_powered_up
    def sw_trigger(self):
        """
        Triggers pulse generation and starts RX transmissions on all
        (master and slave) modules. Should be called only for a master module.
        """
        self.card_handle.SWTrigger()

    @assert_card_is_powered_up
    def set_rx_aperture(self, origin: int, size: int, firing: int=0):
        """
        Sets RX aperture’s origin and size.

        :param origin: an origin channel of the aperture
        :param size: a length of the aperture
        :param firing: a firing, in which the parameter value should apply, starts from 0
        """
        self.log(
            DEBUG,
            "Setting RX aperture: origin=%d, size=%d" % (origin, size)
        )
        self.card_handle.SetRxAperture(origin, size, firing)

    @assert_card_is_powered_up
    def set_rx_time(self, time: float, firing: int=0):
        """
        Sets length of acquisition time.

        :param time: expected acquisition time [s]
        :param firing: a firing, in which the parameter value should apply, starts from 0
        """
        self.log(
            DEBUG,
            "Setting RX time: %f" % time
        )
        self.card_handle.SetRxTime(time, firing)

    @assert_card_is_powered_up
    def set_number_of_firings(self, n_firings: int=1):
        """
        Sets number firings/acquisitions for new TX/RX sequence.
        For each firing/acquisition a different TX/RX parameters can be applied.

        :param n_firings: number of firings to set
        """
        self.log(
            DEBUG,
            "Setting number of firings: %d" % n_firings
        )
        self.card_handle.SetNumberOfFirings(n_firings)

    @assert_card_is_powered_up
    def sw_next_tx(self):
        """
        Sets all TX and RX parameters for the next firing/acquisition.
        """
        self.card_handle.SWNextTX()

    @assert_card_is_powered_up
    def enable_receive(self):
        """
        Enables RX data transfer from the probe’s adapter to the module’s internal memory.
        """
        self.card_handle.EnableReceive()

    @assert_card_is_powered_up
    def enable_transmit(self):
        """
        Enables TX pulse generation.
        """
        self.card_handle.EnableTransmit()

    @assert_card_is_powered_up
    def schedule_receive(self, address, length):
        """
        Schedules a new data transmission from the probe’s adapter to the module’s internal memory.
        This function queues a new data transmission from all available RX channels to the device’s internal memory.
        Data transfer starts with the next “SWTrigger” operation call.

        :param address: module's internal memory address, counted in number of samples
        :param length: number of samples from each channel to acquire
        """
        self.log(
            DEBUG,
            "Scheduling data receive at address=0x%02X, length=%d" % (address, length)
        )
        self.card_handle.ScheduleReceive(
            address*self.dtype.itemsize*self.get_n_rx_channels(),
            self.dtype.itemsize*length*self.get_n_rx_channels()
        )

    @assert_card_is_powered_up
    def set_pga_gain(self, gain):
        """
        Configures programmable-gain amplifier (PGA).

        :param gain: gain to set, available values: 24, 30 [dB]
        :return:
        """
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
        """
        Sets low-pass filter (LPF) cutoff frequency.

        :param cutoff: cutoff frequency to set, available values:
                       10e6, 15e6, 20e6, 30e6, 35e6, 50e6 [Hz]
        """
        self.log(
            DEBUG,
            "Setting LPF Cutoff: %d" % cutoff
        )
        # TODO(pjarosik) dirty, SetLPFCutoff should take values in Hz
        cutoff_mhz = int(cutoff/1e6)
        _utils.assert_true(
            math.isclose(cutoff_mhz*1e6, cutoff),
            "Unavailable LPF cutoff value: %f" % cutoff
        )
        enum_value = self._convert_to_enum_value(
            enum_name="LPF_PROG",
            value=cutoff_mhz,
            unit="MHz"
        )
        self.card_handle.SetLPFCutoff(enum_value)

    @assert_card_is_powered_up
    def set_active_termination(self, active_termination):
        """
        Sets active termination for this card. When active termination is None,
        the property is disabled.

        :param active_termination: active termination, available values: 50, 100,
         200, 400; can be None

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
        """
        Configures low-noise amplifier (LNA) gain.

        :param gain: gain to set; available values: 12, 18, 24 [dB]
        """

        self.log(
            DEBUG,
            "Setting LNA Gain: %d" % gain
        )
        enum_value = self._convert_to_enum_value(
            enum_name="LNA_GAIN_GBL",
            value=gain,
            unit="dB"
        )
        self.card_handle.SetLNAGain(enum_value)

    @assert_card_is_powered_up
    def set_dtgc(self, attenuation):
        """
        Configures time gain compensation (TGC).  When attenuation is None,
        DTGC is set to disabled.

        :param attenuation: attenuation to set, available values: 0, 6, 12,
                            18, 24, 30, 36, 42 [dB]; can be None
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
                att=enum_value
            )
        else:
            self.card_handle.SetDTGC(
                endis=_iarius.EN_DIG_TGC_EN_DIG_TGC_DIS,
                att=_iarius.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_0dB
            )

    @assert_card_is_powered_up
    def enable_test_patterns(self):
        """
        Turns off probe’s RX data acquisition and turns on test patterns generation.
        When test patterns are enabled, sawtooth signal is generated.
        """
        self.log(
            DEBUG,
            "Enabling Test Patterns."
        )
        self.card_handle.EnableTestPatterns()

    @assert_card_is_powered_up
    def disable_test_patterns(self):
        """
        Turns off test patterns generation and turns on probe’s RX data acquisition.
        """
        self.log(
            DEBUG,
            "Disabling Test Patterns."
        )
        self.card_handle.DisableTestPatterns()

    @assert_card_is_powered_up
    def sync_test_patterns(self):
        """
        Waits for update of test patterns.
        """
        self.log(
            DEBUG,
            "Syncing with test patterns..."
        )
        self.card_handle.SyncTestPatterns()

    @assert_card_is_powered_up
    def transfer_rx_buffer_to_host(self, src_addr, length):
        """

        Transfers data from the given module's memory address to the host's
        memory, and returns data buffer (numpy.ndarray).

        **NOTE: This function returns a buffer which is managed internally by the module.
        The content of the buffer may change between successive 'sw_trigger' function calls.
        Please copy buffer's data to your own array before proceeding with the acquisition.**

        :param src_addr: module's memory address, where the RX data was stored.
        :param length: how much data to transfer from each module's channel.
        :return: a buffer (numpy.darray) of shape (length, n_rx_channels), data type: np.int16
        """
        required_nbytes = self.get_n_rx_channels()*length*self.dtype.itemsize
        if self.host_buffer.nbytes != required_nbytes:
            # Intentionally not '<' (we must return an array with given shape).
            self.host_buffer = _utils.create_aligned_array(
                (length, self.get_n_rx_channels()),
                dtype=np.int16,
                alignment=4096
            )
        dst_addr = self.host_buffer.ctypes.data
        length = self.host_buffer.nbytes
        self.log(
            DEBUG,
            "Transferring %d bytes from RX buffer at 0x%02X to host memory at 0x%02X..." % (
                length, src_addr, dst_addr
            )
        )
        _iarius.TransferRXBufferToHostLocation(
            that=self.card_handle,
            dstAddress=dst_addr,
            length=length,
            srcAddress=src_addr # address shift is applied by the low-level layer
        )
        self.log(
            DEBUG,
            "... transferred."
        )
        return self.host_buffer

    def is_powered_down(self):
        """
        Returns true, when module is turned off, false otherwise.

        :return: true, when module is turned off, false otherwise
        """
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