import abc
import numpy as np
from queue import Queue
import logging
from logging import DEBUG, INFO
import time
import arrus.operations as _operations
import arrus.params as _params
import arrus.utils.validation as _validation_util
import arrus.devices.us4oem as _us4oem
import arrus.devices.hv256 as _hv256
import arrus.utils as _utils

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


class AsyncKernel(abc.ABC):

    @abc.abstractmethod
    def stop(self):
        pass


def get_kernel(operation: _operations.Operation, feed_dict: dict):
    device = feed_dict.get("device", None)
    _validation_util.assert_not_none(device, "device")

    # Get kernels dictionary for given device.
    device_type = type(device)
    kernel_registries = {
        _us4oem.Us4OEM: get_us4oem_kernel_registry(),
        _hv256: get_us4oem_kernel_registry()
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


class TxRxModuleKernel(LoadableKernel):

    def __init__(self, op: _operations.TxRx, device: _us4oem.Us4OEM,
                 feed_dict: dict, data_offset=0, sync_required=True,
                 callback=None, firing=0):
        self.op = op
        self.device = device
        self.feed_dict = feed_dict
        self.data_offset = data_offset
        self.sync_required = sync_required
        self.callback = callback
        self.firing = firing

    def load(self):
        self.load_with_sync_option(sync_required=self.sync_required,
                                   callback=self.callback)

    def load_with_sync_option(self, sync_required: bool, callback):
        op = self.op
        device = self.device
        firing = self.firing
        # Tx
        device.set_tx_delays(delays=op.tx.delays, firing=firing)
        # Excitation
        wave = op.tx.excitation
        device.set_tx_frequency(frequency=wave.frequency, firing=firing)
        device.set_tx_half_periods(n_half_periods=int(wave.n_periods*2),
                                   firing=firing)
        device.set_tx_invert(is_enable=wave.inverse, firing=firing)
        # Aperture
        tx_aperture_mask = self._get_aperture_mask(op.tx.aperture, device)
        device.set_tx_aperture_mask(aperture=tx_aperture_mask, firing=firing)
        # RX
        # Aperture
        rx_aperture_mask = self._get_aperture_mask(op.rx.aperture, device)
        device.set_rx_aperture_mask(rx_aperture_mask, firing=firing)
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
            return aperture.mask
        elif isinstance(aperture, _params.RegionBasedAperture):
            mask = np.zeros(device.get_n_channels())
            origin = aperture.origin
            size = aperture.size
            mask[origin:origin+size] = 1
            return mask
        elif isinstance(aperture, _params.SingleElementAperture):
            mask = np.zeros(device.get_n_channels())
            mask[aperture.element] = 1
            return mask
        else:
            raise ValueError("Unsupported aperture type: %s" % type(aperture))

    def validate(self):
        # czy liczba kanalow apertury RX jest 32 - tak jest w tej chwili to realizowane przy transferze danych
        #aperture mask vs liczba wlaczonych kanalow TX, RX (oraz liczba kanalow generalnie)
        pass

    def run(self):
        self.device.start_trigger()
        self.device.enable_receive()
        self.device.trigger_sync()
        result_buffer = _utils.create_aligned_array(
            (self.op.rx.n_samples, self.device.get_n_rx_channels()),
            dtype=np.int16,
            alignment=4096
        )
        self.device.transfer_rx_buffer_to_host_buffer(0, result_buffer)
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

    def validate(self):
        # TODO
        # Czy liczba elementow sekwencji <= 1024
        pass

    def load(self):
        self.load_with_sync_option(sync_required=self.sync_required,
                                   callback=self.callback)

    def load_with_sync_option(self, sync_required: bool, callback):
        # Convert TxRx ops to TxRx kernels
        tx_rx_kernels = []
        data_offset = 0
        n_operations = len(self.op.operations)
        for i, tx_rx in enumerate(self.op.operations):
            sync_required = i == n_operations-1 and sync_required
            kernel = TxRxModuleKernel(op=tx_rx, device=self.device,
                                      feed_dict=self.feed_dict,
                                      data_offset=data_offset,
                                      sync_required=sync_required,
                                      callback=callback, firing=i)
            tx_rx_kernels.append(kernel)
            kernel.load()
            data_offset += tx_rx.rx.n_samples

    def run(self):
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
        buffer = self.device.transfer_rx_buffer_to_host_buffer(0, result_buffer)
        return buffer

    def get_total_n_samples(self):
        return self.total_n_samples

    def _compute_total_n_samples(self):
        total_n_samples = 0
        for tx_rx in self.op.operations:
            total_n_samples += tx_rx.rx.n_samples
        return total_n_samples


class LoopModuleKernel(LoadableKernel, AsyncKernel):

    def __init__(self, op: _operations.Loop, device: _us4oem.Us4OEM,
                 feed_dict: dict):
        self.op = op
        self.device = device
        self.feed_dict = feed_dict
        self.callback = feed_dict.get("callback", None)
        self.data_buffer = None

    def validate(self):
        # Make sure, that callback function was provided
        # Make sure, that operation is Sequence or TxRx
        pass

    def load(self):
        total_n_samples = self.op.get_total_n_samples()

        self.data_buffer = _utils.create_aligned_array(
            (total_n_samples, self.device.get_n_rx_channels()),
            dtype=np.int16,
            alignment=4096
        )

        def _callback_wrapper(event):
            # GIL
            self.device.transfer_rx_buffer_to_host_buffer(0, self.data_buffer)
            self.device.enable_receive()
            self.callback(self.data_buffer)
            # End GIL

        self.op.load_with_sync_option(sync_required=False,
                                      callback=_callback_wrapper)

    def run(self):
        if self.data_queue is None:
            raise ValueError("Run 'load' function first.")
        self.device.start_trigger()
        self.device.enable_receive()
        return self.data_queue

    def stop(self):
        self.device.stop_trigger()
        _logger.info("Stopping the module %s..." % self.device.get_card_id())
        time.sleep(5)

