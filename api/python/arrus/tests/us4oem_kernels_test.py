import unittest
import numpy as np
import dataclasses

from arrus.tests.tools import mock_import
import arrus.validation as _validation


# First, mock packages low-level API libraries.
class IUs4OEMMock:
    pass
class IHV256Mock:
    pass
def GetUs4OEM(index):
    return IUs4OEMMock(index)
class ScheduleReceiveCallbackMock:
    pass
mock_import(
    "arrus.devices.ius4oem",
    IUs4OEM=IUs4OEMMock,
    GetUs4OEM=GetUs4OEM,
    ScheduleReceiveCallback=ScheduleReceiveCallbackMock
)
mock_import(
    "arrus.devices.ihv256",
    IHV256=IHV256Mock
)


from arrus.kernels import (
    TxRxModuleKernel,
    SequenceModuleKernel,
    LoopModuleKernel
)
from arrus.operations import (
    Tx, Rx, TxRx,
    Sequence,
    Loop
)
from arrus.params import (
    SineWave,
    RegionBasedAperture, MaskAperture, SingleElementAperture
)


class Us4OEMDeviceMock:

    def __init__(self):
        N_MAX_FIRINGS = 32
        self.delays = None
        self.tx_frequency = [None]*N_MAX_FIRINGS # just to test sequence
        self.n_half_periods = None
        self.tx_invert = None
        self.tx_aperture_mask = None
        self.rx_aperture_mask = None
        self.rx_time = None
        self.rx_delay = None
        self.schedule_receive_address = None
        self.schedule_receive_total_length = 0
        self.schedule_receive_callback = None
        self.pri = None
        self.state = "stopped"

    def get_n_channels(self):
        return 128

    def get_n_tx_channels(self):
        return 128

    def get_n_rx_channels(self):
        return 32

    def set_tx_delays(self, delays, firing):
        self.delays = delays

    def set_tx_frequency(self, frequency, firing):
        self.tx_frequency[firing] = frequency

    def set_tx_half_periods(self, n_half_periods, firing):
        self.n_half_periods = n_half_periods

    def set_tx_invert(self, is_enable, firing):
        self.tx_invert = is_enable

    def set_tx_aperture_mask(self, aperture, firing):
        self.tx_aperture_mask = aperture

    def set_rx_aperture_mask(self, aperture, firing):
        self.rx_aperture_mask = aperture

    def set_rx_time(self, time, firing):
        self.rx_time = time

    def set_rx_delay(self, delay, firing):
        self.rx_delay = delay

    def schedule_receive(self, address, n_samples, callback):
        self.schedule_receive_address = address
        self.schedule_receive_total_length += n_samples
        self.schedule_receive_callback = callback

    def set_trigger(self, time_to_next_trigger, time_to_next_tx,
                       is_sync_required, idx):
        self.pri = time_to_next_trigger

    def start_trigger(self):
        self.state = "running"

    def stop_trigger(self):
        self.state = "stopped"

    def trigger_sync(self):
        pass

    def enable_receive(self):
        pass

    def transfer_rx_buffer_to_host_buffer(self, src, dst):
        if self.state != "running":
            raise ValueError("Start trigger before running the device")
        dst[:, :] = np.zeros((self.schedule_receive_total_length,
                              self.get_n_rx_channels()))


class TxRxModuleKernelCorrectOperationTest(unittest.TestCase):

    def setUp(self):
        self.device = Us4OEMDeviceMock()
        self.tx = Tx(
            delays=np.linspace(0, 5e-6, 32),
            excitation=SineWave(frequency=5e6, n_periods=2, inverse=False),
            aperture=RegionBasedAperture(64, 32),
            pri=200e-6
        )
        self.rx = Rx(
            sampling_frequency=65e6,
            n_samples=4096,
            aperture=RegionBasedAperture(32, 32)
        )
        self.tx_rx = TxRx(self.tx, self.rx)

    def test_parameters_loaded_correctly(self):
        device = self.device
        tx_rx = self.tx_rx
        tx = self.tx
        rx = self.rx

        TxRxModuleKernel(tx_rx, device, {}).load()

        expected_delays = np.zeros(128)
        expected_delays[64:96] = tx.delays
        np.testing.assert_equal(device.delays, expected_delays)
        self.assertEqual(device.tx_frequency[0], tx.excitation.frequency)
        self.assertEqual(device.n_half_periods, tx.excitation.n_periods*2)
        self.assertEqual(device.tx_invert, tx.excitation.inverse)

        expected_tx_mask = np.zeros(128, dtype=np.float64)
        expected_tx_mask[64:96] = 1.0
        np.testing.assert_equal(device.tx_aperture_mask, expected_tx_mask)
        expected_rx_mask = np.zeros(128, dtype=np.float64)
        expected_rx_mask[32:64] = 1.0
        np.testing.assert_equal(device.rx_aperture_mask, expected_rx_mask)

        self.assertEqual(device.rx_time, rx.rx_time)
        self.assertEqual(device.rx_delay, rx.rx_delay)
        self.assertEqual(device.schedule_receive_address, 0)
        self.assertEqual(device.schedule_receive_total_length, rx.n_samples)
        self.assertIsNone(device.schedule_receive_callback)
        self.assertEqual(device.pri, tx.pri)

    def test_device_run_correctly(self):
        result = TxRxModuleKernel(self.tx_rx, self.device, {}).run()

        self.assertEqual(result.shape,
                         (self.rx.n_samples, self.device.get_n_rx_channels()))


class TxRxModuleKernelValidationTest(unittest.TestCase):

    def setUp(self):
        self.device = Us4OEMDeviceMock()
        self.tx = Tx(
            delays=np.linspace(0, 5e-6, 32),
            excitation=SineWave(frequency=5e6, n_periods=2, inverse=False),
            aperture=RegionBasedAperture(64, 32),
            pri=200e-6
        )
        self.rx = Rx(
            sampling_frequency=65e6,
            n_samples=4096,
            aperture=RegionBasedAperture(32, 32)
        )
        self.tx_rx = TxRx(self.tx, self.rx)

    def test_tx_region_based_aperture_outside_bounds(self):
        to_large_aperture = dataclasses.replace(
            self.tx,
            aperture=RegionBasedAperture(32, 192)
        )
        tx_rx = TxRx(to_large_aperture, self.rx)
        device = self.device
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, device, {}).run()

    def test_tx_mask_aperture_to_large(self):
        large_mask = np.zeros(190)
        large_mask[10] = 1
        wrong_aperture = dataclasses.replace(
            self.tx,
            aperture=MaskAperture(large_mask),
            delays=np.linspace(0, 5e-6, 190)
        )
        tx_rx = TxRx(wrong_aperture, self.rx)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_tx_single_element_aperture_out_of_bounds(self):
        wrong_aperture = dataclasses.replace(
            self.tx,
            aperture=SingleElementAperture(129),
            delays=np.linspace(0, 5e-6, 129)
        )
        tx_rx = TxRx(wrong_aperture, self.rx)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_invalid_number_of_delays(self):
        """
        TX delays array should have the same number of elements as the number
        of active tx aperture channels.
        """
        to_large_aperture = dataclasses.replace(
            self.tx,
            aperture=RegionBasedAperture(0, 66),
            delays=np.linspace(0, 5e-6, 65)
        )
        tx_rx = TxRx(to_large_aperture, self.rx)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_to_large_rx_aperture(self):
        wrong_aperture = dataclasses.replace(
            self.rx,
            aperture=RegionBasedAperture(0, 33)
        )
        tx_rx = TxRx(self.tx, wrong_aperture)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_to_large_rx_aperture(self):
        wrong_aperture = dataclasses.replace(
            self.rx,
            aperture=RegionBasedAperture(0, 33)
        )
        tx_rx = TxRx(self.tx, wrong_aperture)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_rx_aperture_out_of_bounds(self):
        wrong_aperture = dataclasses.replace(
            self.rx,
            aperture=RegionBasedAperture(100, 32)
        )
        tx_rx = TxRx(self.tx, wrong_aperture)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_pri_out_of_range(self):
        wrong_tx = dataclasses.replace(
            self.tx,
            pri=1e-6
        )
        tx_rx = TxRx(wrong_tx, self.rx)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()

    def test_unsupported_sampling_frequency(self):
        wrong_rx = dataclasses.replace(
            self.rx,
            sampling_frequency=100e6
        )
        tx_rx = TxRx(self.tx, wrong_rx)
        with self.assertRaises(_validation.InvalidParameterError):
            TxRxModuleKernel(tx_rx, self.device, {}).run()


class SequenceModuleKernelCorrectTest(unittest.TestCase):

    def setUp(self):
        self.seq = Sequence([
            TxRx(
                tx=Tx(
                    delays=np.linspace(0, 5e-6, 32),
                    excitation=SineWave(frequency=(i+1)*1e6, n_periods=2,
                                        inverse=False),
                    aperture=RegionBasedAperture(64, 32),
                    pri=200e-6
                ),
                rx=Rx(
                    sampling_frequency=65e6,
                    n_samples=4096,
                    aperture=RegionBasedAperture(i*32, 32)
                )
            )
            for i in range(4)
        ])
        self.device = Us4OEMDeviceMock()

    def test_ops_loaded_correctly(self):
        SequenceModuleKernel(self.seq, self.device, {}).load()
        for i in range(4):
            self.assertEqual(self.device.tx_frequency[i], (i+1)*1e6)

    def test_device_run_correctly(self):
        result = SequenceModuleKernel(self.seq, self.device, {}).run()
        total_n_samples = sum([op.rx.n_samples for op in self.seq.operations])
        self.assertEqual(result.shape,
                         (total_n_samples, self.device.get_n_rx_channels()))


if __name__ == "__main__":
    unittest.main()
