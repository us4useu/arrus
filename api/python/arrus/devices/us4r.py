import dataclasses
import numpy as np
import time
import ctypes

import arrus.logging
import arrus.core
from arrus.devices.device import Device, DeviceId, DeviceType
from arrus.ops.us4r import TxRxSequence, Tx, Rx, Pulse
import arrus.exceptions
import arrus.devices.probe
import arrus.metadata

DEVICE_TYPE = DeviceType("Us4R", arrus.core.DeviceType_Us4R)


class FrameChannelMapping:

    def __init__(self, fcm):
        self._fcm_frame = np.zeros(
            (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
            dtype=np.int16)
        self._fcm_channel = np.zeros(
            (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
            dtype=np.int8)
        for frame in range(fcm.getNumberOfLogicalFrames()):
            for channel in range(fcm.getNumberOfLogicalChannels()):
                frame_channel = fcm.getLogical(frame, channel)
                src_frame = frame_channel[0]
                src_channel = frame_channel[1]
                self._fcm_frame[frame, channel] = src_frame
                self._fcm_channel[frame, channel] = src_channel

    @property
    def frames(self):
        return self._fcm_frame

    @property
    def channels(self):
        return self._fcm_channel


class HostBuffer:
    def __init__(
            self,
            buffer_handle,
            fac: arrus.metadata.FrameAcquisitionContext,
            data_description: arrus.metadata.EchoDataDescription,
            frame_shape: tuple
    ):
        """
        :param buffer_handle:
        :param fac:
        :param data_description:
        :param frame_shape: raw frame shape
        """
        self.buffer_handle = buffer_handle
        self.fac = fac
        self.data_description = data_description
        self.frame_shape = frame_shape
        self.buffer_cache = {}

    def tail(self):
        print("Getting tail address")
        data_addr = self.buffer_handle.tailAddress()
        print(f"Data address {data_addr}")
        if data_addr not in self.buffer_cache:
            array = self._create_array(data_addr)
            self.buffer_cache[data_addr] = array
        else:
            array = self.buffer_cache[data_addr]

        # TODO extract first lines from each frame lazily
        metadata = arrus.metadata.Metadata(
            context=self.fac,
            data_desc=self.data_description,
            custom={}
        )
        return array, metadata

    def release_tail(self):
        self.buffer_handle.releaseTail()

    def _create_array(self, addr):
        ctypes_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int16))
        arr = np.ctypeslib.as_array(ctypes_ptr, shape=self.frame_shape)
        return arr

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

    def set_voltage(self, voltage):

        self._handle.setVoltage(voltage)

    def upload(self, seq) -> HostBuffer:
        """
        Uploads a given sequence of operations to perform on the device.

        The host buffer returns Frame Channel Mapping in frame
        acquisition context custom data dictionary.

        :param seq: sequence to set
        :return: a data buffer
        """
        if not isinstance(seq, TxRxSequence):
            # TODO run an appropriate kernel to get a sequence of tx/rx operations
            raise arrus.exceptions.IllegalArgumentError(f"Unhandled operation: {type(seq)}")

        # Convert arrus.ops.us4r.TxRxSequence -> arrus.core.TxRxSequence
        core_seq = arrus.core.TxRxVector()

        # Note: currently only only tx/rx all with the same number of frames is supported.
        n_samples = None

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

            start_sample, end_sample = rx.sample_range
            if n_samples is None:
                n_samples = end_sample - start_sample
            elif n_samples != end_sample - start_sample:
                raise arrus.exception.IllegalArgumentException(
                    "Sequences with the constant number of samples are supported only.")

        core_seq = arrus.core.TxRxSequence(
            sequence=core_seq,
            pri=seq.pri,
            tgcCurve=seq.tgc_curve.tolist())

        upload_result = self._handle.uploadSync(core_seq)

        fcm, buffer_handle = upload_result[0], upload_result[1]
        py_fcm = FrameChannelMapping(fcm)
        print(f"Using fcm: {py_fcm}")

        fac = arrus.metadata.FrameAcquisitionContext(
            device=self._get_dto(),
            sequence=seq,
            medium=self._session.get_session_context().medium,
            custom_data={})

        echo_data_description = arrus.metadata.EchoDataDescription(
            # TODO get fs from device
            # TODO get an array of sampling frequencies for all operations
            sampling_frequency=65e6 / seq.operations[0].rx.downsampling_factor,
            custom={
                "frame_channel_mapping": py_fcm
            }
        )
        return HostBuffer(
            buffer_handle=buffer_handle,
            fac=fac, data_description=echo_data_description,
            frame_shape=self._get_physical_frame_shape(py_fcm, n_samples))

    def _get_physical_frame_shape(self, fcm, n_samples, n_channels=32):
        # NOTE: We assume here, that each frame has the same number of samples!
        # This might not be case in further improvements.
        n_frames = np.max(fcm.frames) + 1
        print(n_frames)
        print(f"max frame: {n_frames}")
        return n_frames * n_samples, n_channels

    def _get_dto(self):
        arrus.logging.log(arrus.logging.DEBUG, "producing dto")
        n_elements = arrus.core.getNumberOfElements(self._handle.getProbe(0).getModel())
        arrus.logging.log(arrus.logging.DEBUG, "getting a pitch")
        pitch = arrus.core.getPitch(self._handle.getProbe(0).getModel())
        curvature_radius = self._handle.getProbe(0).getModel().getCurvatureRadius()
        arrus.logging.log(arrus.logging.DEBUG, "After getting probe data")
        probe_model = arrus.devices.probe.ProbeModel(
            n_elements=n_elements,
            pitch=pitch,
            curvature_radius=curvature_radius)
        probe_dto = arrus.devices.probe.ProbeDTO(model=probe_model)
        return Us4RDTO(probe=probe_dto)

    def start(self):
        self._handle.start()

    def stop(self):
        self._handle.stop()

    def disable_hv(self):
        self._handle.disableHV()

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
        self.i = (i + 1) % self.n_frames
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
