import abc
import numpy as np
from queue import Queue
import logging
from logging import DEBUG, INFO
import time
import arrus.operations as _operations
import arrus.params as _params
import arrus.devices.us4oem as _us4oem
import arrus.devices.hv256 as _hv256
import arrus.utils as _utils
import arrus.validation as _validation

_logger = logging.getLogger(__name__)


class Kernel(abc.ABC):

    @abc.abstractmethod
    def run(self):
        pass


class LoadableKernel(Kernel):

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def run_loaded(self):
        pass

    def run(self):
        self.validate()
        self.load()
        return self.run_loaded()


class AsyncKernel(abc.ABC):

    @abc.abstractmethod
    def stop(self):
        pass


def get_kernel(operation: _operations.Operation, feed_dict: dict):
    device = feed_dict.get("device", None)
    _validation.assert_not_none(device, "device")

    # Get kernels dictionary for given device.
    device_type = type(device)
    kernel_registries = {
        _us4oem.Us4OEM: get_us4oem_kernel_registry(),
        _hv256: get_hv256_kernel_registry()
    }
    if device_type not in kernel_registries:
        raise ValueError("Unsupported device type: %s" % device_type)
    kernel_registry = kernel_registries[device_type]

    # Get operation's kernel from the registry.
    operation_type = type(operation)
    if operation_type not in kernel_registry:
        raise ValueError("Unsupported operation '%s' for device type: %s."%
                         (operation_type, device_type))
    kernel = kernel_registry[operation_type](operation, device, feed_dict)
    return kernel


def get_us4oem_kernel_registry():
    return {
        _operations.TxRx: TxRxModuleKernel,
        _operations.Sequence: SequenceModuleKernel,
        _operations.Loop: LoopModuleKernel
    }


def get_hv256_kernel_registry():
    return {
        _operations.SetHVVoltage: SetHVVoltageKernel,
        _operations.DisableHVVoltage: DisableHVVoltageKernel,
    }


class TxRxModuleKernel(LoadableKernel):

    def __init__(self, op: _operations.TxRx, device: _us4oem.Us4OEM,
                 feed_dict: dict, data_offset=0, sync_required=True,
                 callback=None, firing=0):
        self.op = op
        self.device = device
        self.feed_dict = feed_dict
        self.data_offset = data_offset
        self._sync_required = sync_required
        self._callback = callback
        self.firing = firing
        self._tx_aperture_mask = self._get_aperture_mask(op.tx.aperture, device)
        self._rx_aperture_mask = self._get_aperture_mask(op.rx.aperture, device)

    def load(self):
        self.load_with_sync_option(sync_required=self._sync_required,
                                   callback=self._callback)

    def load_with_sync_option(self, sync_required: bool, callback):
        op = self.op
        device = self.device
        firing = self.firing
        # Tx
        tx_delays = np.zeros(device.get_n_tx_channels())
        tx_delays[np.where(self._tx_aperture_mask)] = op.tx.delays
        device.set_tx_delays(delays=tx_delays, firing=firing)
        # Excitation
        wave = op.tx.excitation
        device.set_tx_frequency(frequency=wave.frequency, firing=firing)
        device.set_tx_half_periods(n_half_periods=int(wave.n_periods*2),
                                   firing=firing)
        device.set_tx_invert(is_enable=wave.inverse, firing=firing)
        # Aperture
        device.set_tx_aperture_mask(aperture=self._tx_aperture_mask,
                                    firing=firing)
        # RX
        # Aperture
        device.set_rx_aperture_mask(aperture=self._rx_aperture_mask,
                                    firing=firing)
        # Samples, rx time, delay
        n_samples = op.rx.n_samples
        device.set_rx_time(time=op.rx.rx_time, firing=firing)
        device.set_rx_delay(delay=op.rx.rx_delay, firing=firing)
        device.schedule_receive(self.data_offset, n_samples, callback)
        device.set_trigger(time_to_next_trigger=op.tx.pri, time_to_next_tx=0,
                           is_sync_required=sync_required, idx=firing)

    @staticmethod
    def _get_aperture_mask(aperture, device: _us4oem.Us4OEM):
        if isinstance(aperture, _params.MaskAperture):
            return aperture.mask.astype(bool)
        elif isinstance(aperture, _params.RegionBasedAperture):
            mask = np.zeros(device.get_n_channels()).astype(bool)
            origin = aperture.origin
            size = aperture.size
            mask[origin:origin+size] = True
            return mask
        elif isinstance(aperture, _params.SingleElementAperture):
            mask = np.zeros(device.get_n_channels()).astype(bool)
            _validation.assert_in_range(aperture.element,
                                        (0, device.get_n_channels()),
                                        "single element aperture")
            mask[aperture.element] = True
            return mask
        else:
            raise ValueError("Unsupported aperture type: %s" % type(aperture))

    def validate(self):
        device = self.device
        # TX
        tx_aperture = self.op.tx.aperture
        self._validate_aperture(tx_aperture, device.get_n_channels(), "tx")
        # Delays:
        number_of_active_elements = np.sum(self._tx_aperture_mask.astype(bool))
        _validation.assert_shape(self.op.tx.delays,
                                 (number_of_active_elements,),
                                 parameter_name="tx.delays")

        # Excitation:
        excitation = self.op.tx.excitation
        _validation.assert_type(excitation, _params.SineWave, "tx.excitation")
        _validation.assert_in_range(excitation.frequency, (1e6, 10e6),
                                    "tx.excitation.frequency")
        _validation.assert_in_range(excitation.n_periods, (1, 20),
                                    "tx.excitation.n_periods")
        # Triggers:
        _validation.assert_in_range(self.op.tx.pri, (100e-6, 2000e-6),
                                    "tx.pri")
        # RX
        rx_aperture = self.op.rx.aperture
        self._validate_aperture(rx_aperture, device.get_n_channels(), "rx")
        rx_aperture_mask = self._rx_aperture_mask.astype(bool)
        _validation.assert_in_range(np.sum(rx_aperture_mask), (0, 32),
                                    "rx.aperture number of channels")
        _validation.assert_in_range(self.op.rx.n_samples,
                                    (0, 65536), "rx.n_samples")
        _validation.assert_one_of(self.op.rx.sampling_frequency,
                                  {32.5e6, 65e6}, "rx.sampling_frequency")

    def _validate_aperture(self, aperture, n_channels, aperture_type):
        # TODO(pjarosik) this validation should be performed before creating
        #  aperture masks
        if isinstance(aperture, _params.MaskAperture):
            expected_shape = (n_channels,)
            _validation.assert_shape(aperture.mask, expected_shape,
                                     "%s.aperture" % aperture_type)
        elif isinstance(aperture, _params.RegionBasedAperture):
            start = aperture.origin
            end = aperture.origin + aperture.size
            _validation.assert_in_range((start, end), (0, n_channels),
                                        parameter_name="%s.aperture" %
                                                       aperture_type)
        elif isinstance(aperture, _params.SingleElementAperture):
            channel = aperture.element
            _validation.assert_in_range((channel, channel), (0, n_channels),
                                        parameter_name="%s.aperture" %
                                                       aperture_type)
        else:
            raise ValueError("Unsupported aperture type: %s" % type(aperture))

    def run_loaded(self):
        self.device.start_trigger()
        self.device.enable_receive()
        self.device.trigger_sync()
        result_buffer = _utils.create_aligned_array(
            (self.op.rx.n_samples, self.device.get_n_rx_channels()),
            dtype=np.int16,
            alignment=4096
        )
        self.device.transfer_rx_buffer_to_host_buffer(0, result_buffer)
        self.device.stop_trigger()
        return result_buffer

    def get_total_n_samples(self):
        return self.op.rx.n_samples


class SequenceModuleKernel(LoadableKernel):

    def __init__(self, op: _operations.Sequence, device: _us4oem.Us4OEM,
                 feed_dict: dict, sync_required=True, callback=None):
        self.op = op
        self.device = device
        self.feed_dict = feed_dict
        self.total_n_samples = self._compute_total_n_samples()
        self.sync_required = sync_required
        self.callback = callback
        self._kernels = self._get_txrx_kernels(self.op.operations,
                                               self.feed_dict,
                                               self.device, sync_required,
                                               self.callback)

    def validate(self):
        seq = self.op
        _validation.assert_not_greater_than(len(seq.operations), 1024,
                                            "number of operations in sequence")
        for tx_rx in self._kernels:
            tx_rx.validate()

    def load(self):
        for kernel in self._kernels:
            kernel.load()

    def run_loaded(self):
        self.device.start_trigger()
        self.device.enable_receive()
        self.device.trigger_sync()
        if self.total_n_samples is None:
            raise ValueError("Call 'load' function first.")

        result_buffer = _utils.create_aligned_array(
            (self.total_n_samples, self.device.get_n_rx_channels()),
            dtype=np.int16,
            alignment=4096
        )
        self.device.transfer_rx_buffer_to_host_buffer(0, result_buffer)
        return result_buffer

    def get_total_n_samples(self):
        return self.total_n_samples

    def _compute_total_n_samples(self):
        total_n_samples = 0
        for tx_rx in self.op.operations:
            total_n_samples += tx_rx.rx.n_samples
        return total_n_samples

    def _get_txrx_kernels(self, operations, feed_dict, device, sync_required,
                          callback):
        tx_rx_kernels = []
        data_offset = 0
        n_operations = len(operations)
        for i, tx_rx in enumerate(operations):
            sync_required = i == n_operations-1 and sync_required
            if i == n_operations-1:
                cb = callback
            else:
                cb = None
            kernel = TxRxModuleKernel(op=tx_rx, device=device,
                                      feed_dict=feed_dict,
                                      data_offset=data_offset,
                                      sync_required=sync_required,
                                      callback=cb, firing=i)
            tx_rx_kernels.append(kernel)
            data_offset += tx_rx.rx.n_samples
        return tx_rx_kernels


class LoopModuleKernel(LoadableKernel, AsyncKernel):

    def __init__(self, op: _operations.Loop, device: _us4oem.Us4OEM,
                 feed_dict: dict):
        self.op = op
        self.device = device
        self.feed_dict = feed_dict
        self._data_buffer = self._create_data_buffer(self.op)
        self._callback = self._create_callback(feed_dict, self._data_buffer)
        self._kernel = self._create_kernel(self.op, self.device,
                                           self.feed_dict, self._callback)

    def validate(self):
        self._kernel.validate()

    def load(self):
        self._kernel.load()

    def run_loaded(self):
        self.device.start_trigger()
        self.device.enable_receive()
        return None

    def stop(self):
        self.device.stop_trigger()
        _logger.log(INFO, "Waiting for module to stop %s..." %
                    self.device.get_id())
        time.sleep(5)
        _logger.log(INFO, "... module stopped.")

    def _create_kernel(self, op, device, feed_dict, callback):
        if isinstance(op, _operations.TxRx):
            return TxRxModuleKernel(op, device, feed_dict, sync_required=False,
                                    callback=callback)
        elif isinstance(op, _operations.Sequence):
            return SequenceModuleKernel(op, device, feed_dict,
                                        sync_required=False, callback=callback)
        else:
            raise ValueError("Invalid type of operation to perform in loop, "
                             "should be one of: %s, %s" % (_operations.TxRx,
                                                           _operations.Sequence)
                             )

    def _create_data_buffer(self, op):
        if isinstance(op, _operations.TxRx):
            return _operations.TxRx.rx.n_samples
        elif isinstance(op, _operations.Sequence):
            return sum([txrx.rx.n_samples for txrx in op.operations])
        else:
            return None

    def _create_callback(self, feed_dict, data_buffer):
        cb = feed_dict.get("callback", None)
        _validation.assert_not_none(cb, "callback")

        def _callback_wrapper(event):
            # GIL
            self.device.transfer_rx_buffer_to_host_buffer(0, data_buffer)
            self.device.enable_receive()
            cb(data_buffer)
            # End GIL

        return _callback_wrapper


class SetHVVoltageKernel(Kernel):

    def __init__(self, op: _operations.SetHVVoltage, device: _hv256.HV256,
                 feed_dict: dict):
        self.op = op
        self.hv256 = device
        self.feed_dict = feed_dict

    def run(self):
        # TODO validate
        self.hv256.enable_hv()
        self.hv256.set_hv_voltage(self.op.voltage)


class DisableHVVoltageKernel(Kernel):

    def __init__(self, op: _operations.DisableHVVoltage, device: _hv256.HV256,
                 feed_dict: dict):
        self.op = op
        self.hv256 = device
        self.feed_dict = feed_dict

    def run(self):
        self.hv256.disable_hv()

