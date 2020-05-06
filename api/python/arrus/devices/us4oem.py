import math
import time
from functools import wraps
from logging import DEBUG, INFO
from typing import List

import numpy as np

import arrus.devices.device as _device
import arrus.devices.ius4oem as _ius4oem
import arrus.devices.callbacks as _callbacks
import arrus.utils as _utils


def assert_card_is_powered_up(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        us4oem = args[0]
        if us4oem.is_powered_down():
            raise RuntimeError("Card is powered down. Start the card first.")
        return f(*args, **kwargs)
    return wrapper


class Us4OEM(_device.Device):
    """
    A single Us4OEM.
    """
    _DEVICE_NAME = "Us4OEM"

    @staticmethod
    def get_card_id(index):
        return _device.Device.get_device_id(Us4OEM._DEVICE_NAME, index)

    def __init__(self, index: int, card_handle: _ius4oem.IUs4OEM):
        super().__init__(Us4OEM._DEVICE_NAME, index)
        self.card_handle = card_handle
        self.dtype = np.dtype(np.int16)
        self.host_buffer = np.array([])
        self.pri_list = None
        self.pri_total = None
        self.callbacks = []

    def start_if_necessary(self):
        """
        Starts the card if is powered down.
        """
        if self.is_powered_down():
            self.log(
                INFO,
                "Was powered down, initializing it and powering up...")

            self.card_handle.Initialize()
            self.set_tx_channel_mapping(self.tx_channel_mapping)
            self.set_rx_channel_mapping(self.rx_channel_mapping)
            self.log(INFO, "... successfully powered up.")

    def get_n_rx_channels(self):
        """
        Returns number of RX channels.

        :return: number of RX channels.
        """
        return 32 # TODO(pjarosik) should be returned by us4oem
        # return self.card_handle.GetNRxChannels()

    def get_n_tx_channels(self):
        """
        Returns number of TX channels.

        :return: number of TX channels
        """
        return 128 # TODO(pjarosik) should be returned by ius4oem
        # return self.card_handle.GetNTxChannels()

    def get_n_channels(self):
        """
        Returns number of channels available for this module.

        :return: number of channels of the module
        """
        return 128 # TODO(pjarosik) should be returned by ius4oem

    def store_mappings(self, tx_m, rx_m):
        self.tx_channel_mapping = tx_m
        self.rx_channel_mapping = rx_m

    @assert_card_is_powered_up
    def set_tx_channel_mapping(self, tx_channel_mapping: List[int]):
        """
        Sets card's TX channel mapping.

        :param tx_channel_mapping: a list, where list[interface channel] = us4oem channel
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

        :param rx_channel_mapping: a list, where list[interface channel] = us4oem channel
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
    def set_tx_aperture_mask(self, aperture, firing: int=0):
        """
        Sets mask of active TX aperture channels.

        :param aperture: a boolean numpy array of get_number_of_channels() channels, 'True' means to activate chanel on a given position.
        :param firing: a firing, in which the delay should apply, **starts from 0**
        """
        if isinstance(aperture, list):
            aperture = np.array(aperture).astype(np.bool)
        if len(aperture.shape) != 1:
            raise ValueError("Aperture should be a vector.")
        if aperture.shape[0] != self.get_n_channels():
            raise ValueError("Aperture should have %d elements."
                             % aperture.shape[0])
        if aperture.dtype != np.bool:
            raise ValueError("Aperture should be a vector of booleans.")
        aperture = aperture.astype(np.uint16)
        aperture_list = aperture.tolist()
        n = len(aperture_list)
        self.log(DEBUG, "Setting TX aperture: %s" % str(aperture_list))
        array = None
        try:
            array = _ius4oem.new_uint16Array(n)
            for i, sample in enumerate(aperture_list):
                _ius4oem.uint16Array_setitem(array, i, sample)
            _ius4oem.setTxApertureCustom(self.card_handle, array, n, firing)
        finally:
            _ius4oem.delete_uint16Array(array)

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
    def set_rx_aperture_mask(self, aperture, firing: int = 0):
        """
        Sets position of an active RX aperture.

        :param aperture: a boolean numpy array of get_number_of_channels() channels, 'True' means to activate chanel on a given position.
        :param firing: a firing, in which the delay should apply, **starts from 0**
        """
        if isinstance(aperture, list):
            aperture = np.array(aperture).astype(np.bool)
        if len(aperture.shape) != 1:
            raise ValueError("Aperture should be a vector.")
        if aperture.shape[0] != self.get_n_channels():
            raise ValueError("Aperture should have %d elements."
                             % aperture.shape[0])
        if aperture.dtype != np.bool:
            raise ValueError("Aperture should be a vector of booleans.")
        aperture = aperture.astype(np.uint16)
        aperture_list = aperture.tolist()
        n = len(aperture_list)
        self.log(DEBUG, "Setting TX aperture: %s" % str(aperture_list))
        array = None
        try:
            array = _ius4oem.new_uint16Array(n)
            for i, sample in enumerate(aperture_list):
                _ius4oem.uint16Array_setitem(array, i, sample)
            _ius4oem.setRxApertureCustom(self.card_handle, array, n, firing)
        finally:
            _ius4oem.delete_uint16Array(array)

    @assert_card_is_powered_up
    def set_tx_delays(self, delays, firing: int=0):
        """
        Sets channel's TX delays.

        :param delays: an array of delays to set [s], number of elements
                       should be equal to the number of module's TX channels
        :param firing: a firing, in which the delay should apply, **starts from 0**
        :return: an array of values, which were set [s]
        """
        if isinstance(delays, np.ndarray):
            if len(delays.shape) != 1:
                raise ValueError("Delays should be a vector.")
            if delays.shape[0] != self.get_n_tx_channels():
                raise ValueError("You should provide %d delays." % self.get_n_tx_channels())
            delays = delays.tolist()
        _utils._assert_equal(
            len(np.squeeze(delays)), self.get_n_tx_channels(),
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
    def set_rx_delay(self, delay: float, firing: int=0):
        """
        Sets the starting point of the acquisition time [s]

        :param delay: expected acquisition time starting point relative to trigger [s]
        :param firing: an firing, in which the parameter value should apply, **starts from 0**
        """
        self.log(DEBUG, "Setting RX delay = %f for firing %d" % (delay, firing))
        self.card_handle.SetRxDelay(delay=delay, firing=firing)

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
    def set_tx_half_periods(self, n_half_periods: int, firing: int=0):
        """
        Sets number of TX signal half-periods.

		:param n_half_periods: number of half-periods to set
		:param firing: a firing, in which the parameter value should apply, **starts from 0**
		:return: an exact number of half-periods that has been set on a given module
        """
        self.log(
            DEBUG,
            "Setting number of half periods: %f" % n_half_periods
        )
        return self.card_handle.SetTxHalfPeriods(n_half_periods, firing)

    @assert_card_is_powered_up
    def sw_trigger(self):
        """
        Triggers pulse generation and starts RX transmissions on all
        (master and slave) modules. Should be called only for a master module.
        """
        self.card_handle.SWTrigger()

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
        _ius4oem.EnableReceiveDelayed(self.card_handle)

    @assert_card_is_powered_up
    def enable_transmit(self):
        """
        Enables TX pulse generation.
        """
        self.card_handle.EnableTransmit()

    @assert_card_is_powered_up
    def schedule_receive(self, address, length, callback=None):
        """
        Schedules a new data transmission from the probe’s adapter to the module’s internal memory.
        This function queues a new data transmission from all available RX channels to the device’s internal memory.
        Data transfer starts with the next “SWTrigger” operation call.

        :param address: module's internal memory address, counted in number of samples
        :param length: number of samples from each channel to acquire
        :param: callback: a callback function to call when data become available.
        The callback function should take one parameter of type DataAcquiredEvent.
        """
        self.log(
            DEBUG,
            "Scheduling data receive at address=0x%02X, length=%d" % (address, length)
        )

        address = address*self.dtype.itemsize*self.get_n_rx_channels()
        length = self.dtype.itemsize*length*self.get_n_rx_channels()
        if callback is None:
            _ius4oem.ScheduleReceiveWithoutCallback(
                self.card_handle,
                address=address,
                length=length
            )
        else:
            cbk = _callbacks.ScheduleReceiveCallback(callback)
            self.callbacks.append(cbk)
            _ius4oem.ScheduleReceiveWithCallback(
                self.card_handle,
                address=address,
                length=length,
                callback=cbk
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
        if active_termination is not None:
            self.log(
                DEBUG,
                "Setting active termination: %d" % active_termination
            )
            enum_value = self._convert_to_enum_value(
                enum_name="GBL_ACTIVE_TERM",
                value=active_termination,
            )
            self.card_handle.SetActiveTermination(
                endis=_ius4oem.ACTIVE_TERM_EN_ACTIVE_TERM_EN,
                term=enum_value
            )
        else:
            self.log(DEBUG, "Disabling active termination.")
            self.card_handle.SetActiveTermination(
                endis=_ius4oem.ACTIVE_TERM_EN_ACTIVE_TERM_DIS,
                # TODO(pjarosik) when disabled, what value should be set?
                term=_ius4oem.GBL_ACTIVE_TERM_GBL_ACTIVE_TERM_50
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

        if attenuation is not None:
            self.log(
                DEBUG,
                "Setting DTGC: %d" % attenuation
            )
            enum_value = self._convert_to_enum_value(
                enum_name="DIG_TGC_ATTENUATION",
                value=attenuation,
                unit="dB"
            )
            self.card_handle.SetDTGC(
                endis=_ius4oem.EN_DIG_TGC_EN_DIG_TGC_EN,
                att=enum_value
            )
        else:
            self.log(DEBUG, "Disabling DTGC")
            self.card_handle.SetDTGC(
                endis=_ius4oem.EN_DIG_TGC_EN_DIG_TGC_DIS,
                att=_ius4oem.DIG_TGC_ATTENUATION_DIG_TGC_ATTENUATION_0dB
            )

    @assert_card_is_powered_up
    def enable_tgc(self):
        """
        Enables Time Gain Compensation (TGC).
        """
        self.card_handle.TGCEnable()
        self.log(DEBUG, "TGC enabled.")

    @assert_card_is_powered_up
    def disable_tgc(self):
        """
        Disables Time Gain Compensation (TGC).
        """
        self.card_handle.TGCDisable()
        self.log(DEBUG, "TGC disabled.")

    @assert_card_is_powered_up
    def set_tgc_samples(self, samples):
        """
        Sets TGC samples.

        :param samples: TGC samples to set
        """
        if isinstance(samples, np.ndarray):
            if len(samples.shape) != 1:
                raise ValueError("'Samples' should be a vector.")
            samples = samples.tolist()

        self.log(DEBUG, "Setting TGC samples: %s" % str(samples))
        n = len(samples)
        array = None
        try:
            array = _ius4oem.new_uint16Array(n)
            for i, sample in enumerate(samples):
                _ius4oem.uint16Array_setitem(array, i, sample)
            self.card_handle.TGCSetSamples(array, n)
        finally:
            _ius4oem.delete_uint16Array(array)

    @assert_card_is_powered_up
    def set_tx_invert(self, is_enable: bool, firing: int=0):
        """
        Enables/disables inversion of TX signal.

        :param is_enable: should the inversion be enabled?
        :param firing: a firing, in which the parameter value should apply
        """
        if is_enable:
            self.log(DEBUG, "Enabling inversion of TX signal.")
        else:
            self.log(DEBUG, "Disabling inversion of TX signal.")
        self.card_handle.SetTxInvert(onoff=is_enable, firing=firing)

    @assert_card_is_powered_up
    def set_tx_cw(self, is_enable: bool, firing: int=0):
        """
        Enables/disables generation of long TX bursts.

        :param is_enable: should the generation of long TX bursts be enabled?
        :param firing: a firing, in which the parameter value should apply
        """
        if is_enable:
            self.log(DEBUG, "Enabling generation of long TX bursts.")
        else:
            self.log(DEBUG, "Disabling geneartion of long TX bursts.")
        self.card_handle.SetTxCw(onoff=is_enable, firing=firing)

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
        _ius4oem.TransferRXBufferToHostLocation(
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

    @assert_card_is_powered_up
    def clear_scheduled_receive(self):
        """
        Clears scheduled receive queue.
        """
        self.log(
            DEBUG,
            "Clearing scheduled receive requests..."
        )
        self.card_handle.ClearScheduledReceive()

    @assert_card_is_powered_up
    def trigger_start(self):
        """
        Starts generation of the hardware trigger.
        """
        self.log(
            DEBUG,
            "Starting generation of the hardware trigger..."
        )
        if self.pri_list is None and self.pri_total is None:
            raise ValueError("Call 'set_n_triggers' first")
        self.pri_total = sum(self.pri_list)
        self.pri_list = None
        self.card_handle.TriggerStart()
        time.sleep(self.pri_total)

    @assert_card_is_powered_up
    def trigger_stop(self):
        """
        Stops generation of the hardware trigger.
        """
        self.log(
            DEBUG,
            "Stops generation of the hardware trigger..."
        )
        self.card_handle.TriggerStop()

    @assert_card_is_powered_up
    def trigger_sync(self):
        """
        Resumes generation of the hardware trigger.
        """
        self.log(
            DEBUG,
            "Resuming generation of the hardware trigger..."
        )
        if self.pri_total is None:
            raise ValueError("Call 'trigger_start' first.")
        self.card_handle.TriggerSync()
        time.sleep(self.pri_total)

    @assert_card_is_powered_up
    def set_n_triggers(self, n_triggers):
        """
        Sets the number of trigger to be generated.

        :param n_triggers: number of triggers to set

        """
        self.log(
            DEBUG,
            "Setting number of triggers to generate to %d..." % n_triggers
        )
        # TODO(pjarosik) trigger_sync should wait until all data is available
        self.pri_total = None
        self.pri_list = [0.0]*n_triggers
        self.card_handle.SetNTriggers(n_triggers)

    @assert_card_is_powered_up
    def set_trigger(self,
                    time_to_next_trigger: float,
                    time_to_next_tx: float=0.0,
                    is_sync_required: bool=False,
                    idx: int=0):
        """
        Sets parameters of the trigger event.
        Each trigger event will generate a trigger signal for the current
        firing/acquisition and set next firing parameters.

        :param timeToNextTrigger: time between current and the next trigger [s]
        :param timeToNextTx: delay between current trigger and setting next firing parameters [s]
        :param syncReq: should the trigger generator pause and wait for the trigger_sync() call
        :param idx: a firing, in which the parameters values should apply, **starts from 0**
        """

        #TODO(pjarosik) dirty, should be handled by an IUs4OEM implementation
        time_to_next_trigger_us = int(time_to_next_trigger*1e6)
        if not math.isclose(time_to_next_trigger_us/1e6, time_to_next_trigger):
            raise RuntimeError(
                "Numeric error when computing time to next trigger, "
                "input value %.10f [s], value to set %.10f [us]."
                    %(time_to_next_trigger, time_to_next_trigger_us)
            )

        time_to_next_tx_us = int(time_to_next_tx*1e6)
        if not math.isclose(time_to_next_tx_us/1e6, time_to_next_tx):
            raise RuntimeError(
                "Numeric error when computing time to next TX, "
                "input value %.10f [s], value to set %.10f [us]."
                    %(time_to_next_tx, time_to_next_tx_us)
            )

        self.log(
            DEBUG,
            ("Setting trigger generation parameters to: "
             "trigger number: %d, "
             "time to next trigger: %f [s] (%d [us]), "
             "time to next tx: %f [s] (%d [us]), "
             "is sync required: %s") %
            (idx,
             time_to_next_trigger, time_to_next_trigger_us,
             time_to_next_tx, time_to_next_tx_us,
             str(is_sync_required))
        )
        self.card_handle.SetTrigger(
            timeToNextTrigger=time_to_next_trigger_us,
            timeToNextTx=time_to_next_tx,
            syncReq=is_sync_required,
            idx=idx
        )
        self.pri_list[idx] = time_to_next_trigger

    def _convert_to_enum_value(self, enum_name, value, unit=""):
        _utils.assert_true(
            round(value) == value,
            "Value %s for '%s' should be an integer value." % (value, enum_name)
        )
        const_prefix = enum_name + "_" + enum_name
        const_name = const_prefix + "_" + str(value) + unit
        try:
            return getattr(_ius4oem, const_name)
        except AttributeError:
            acceptable_values = set()
            for key in dir(_ius4oem):
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

