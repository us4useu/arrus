import dataclasses
import logging
import numpy as np
import time

import arrus.core
from arrus.devices.device import Device, DeviceId, DeviceType
from arrus.ops.us4r import TxRxSequence, Tx, Rx, Pulse
import arrus.exceptions
import arrus.devices.probe
import arrus.metadata


DEVICE_TYPE = DeviceType("Us4R", arrus.core.DeviceType_Us4R)


class FrameChannelMapping:
    pass


class HostBuffer:

    def __init__(
            self,
            buffer_handle,
            fac: arrus.metadata.FrameAcquisitionContext,
            fcm: FrameChannelMapping,
            data_description: arrus.metadata.EchoDataDescription):
        self.buffer_handle = buffer_handle
        self.fac = fac
        self.fcm = fcm
        self.data_description = data_description

    def tail(self):
        # TODO buffer_handle.tail()
        # TODO wrap into numpy array
        # TODO wrap Context into Metadata class
        pass

    def release_tail(self):
        pass

class Us4R(Device):
    """
    A handle to Us4R device.
    """
    def __init__(self, handle, parent_session):
        super().__init__()
        self._handle = handle
        self._session = parent_session
        self._device_id = DeviceId(
            DEVICE_TYPE,
            self._handle.getDeviceId().getOrdinal())

    def get_device_id(self):
        return self._device_id

    def upload(self, seq):
        if not isinstance(seq, TxRxSequence):
            # TODO run an appropriate kernel to get a sequence of tx/rx operations
            raise arrus.exceptions.IllegalArgumentError(
                f"Unhandled operation: {type(seq)}")
        # Transfer TxRxSequence -> arrus.core.TxRxSequence

        core_seq = arrus.core.TxRxVector()
        for op in seq.operations:
            tx, rx = op.tx, op.rx
            # TODO validate shape

            # TX
            core_delays = np.zeros(tx.aperture.shape, dtype=np.float32)
            core_delays[tx.aperture] = tx.delays
            core_excitation = arrus.core.Pulse(
                centerFrequency=tx.excitation.center_frequency,
                nPeriods=tx.excitation.n_periods,
                inverse=tx.excitation.inverse
            )
            core_tx = arrus.core.Tx(
                aperture=arrus.core.VectorBool(tx.aperture.tolist()),
                delays=arrus.core.VectorFloat(core_delays.tolist()),
                excitation=core_excitation
            )
            # RX
            core_rx = arrus.core.Rx(
                arrus.core.VectorBool(rx.aperture.tolist()),
                arrus.core.PairUint32(rx.sample_range[0], rx.sample_range[1]),
                rx.downsampling_factor,
                arrus.core.PairChannelIdx(rx.padding[0], rx.padding[1])
            )
            core_txrx = arrus.core.TxRx(core_tx, core_rx)
            arrus.core.TxRxVectorPushBack(core_seq, core_txrx)

        core_seq = arrus.core.TxRxSequence(
            sequence=core_seq,
            pri=seq.pri,
            tgcCurve=seq.tgc_curve.tolist()
        )
        self._handle.upload(core_seq)

        # fac = arrus.metadata.FrameAcquisitionContext(
        #     # convert handle.get_probe().probeModel to ProbeDTO
        #     device=self.get_dto(),
        #     sequence=seq,
        #     medium=self._session.get_session_context(),
        #     custom_data={})

        # upload sequence
        # wrap frame channel mapping
        # add fcm to context, add context to constant metadata, return buffer

    def start(self):
        self._handle.start()

    def stop(self):
        self._handle.stop()

    def set_voltage(self, voltage):
        self._handle.set_voltage(voltage)

    def disable_hv(self):
        self._handle.disable_hv()


# ------------------------------------------ LEGACY MOCK
class MockFileBuffer:
    def __init__(self, dataset: np.ndarray, metadata):
        self.dataset = dataset
        self.n_frames, _, _, _ = dataset.shape
        self.i = 0
        self.counter = 0
        self.metadata = metadata

    def pop(self):
        i = self.i
        self.i = (i+1) % self.n_frames
        custom_data = {
            "pulse_counter": self.counter,
            "trigger_counter": self.counter,
            "timestamp": time.time_ns() // 1000000
        }
        self.counter += 1

        metadata = arrus.metadata.Metadata(
            context=self.metadata.context,
            data_desc=self.metadata.data_description,
            custom=custom_data)
        return np.array(self.dataset[self.i, :, :, :]), metadata


class MockUs4R(Device):
    def __init__(self, dataset: np.ndarray, metadata, index: int):
        super().__init__("Us4R", index)
        self.dataset = dataset
        self.metadata = metadata
        self.buffer = None

    def upload(self, sequence):
        self.buffer = MockFileBuffer(self.dataset, self.metadata)
        return self.buffer

    def start(self):
        pass

    def stop(self):
        pass

    def set_hv_voltage(self, voltage):
        pass

    def disable_hv(self):
        pass

@dataclasses.dataclass(frozen=True)
class Us4RDTO:
    probe: arrus.devices.probe.ProbeDTO

    def get_id(self):
        return "Us4R:0"
