import abc
import numpy as np
import importlib
import dataclasses

import arrus.core
import arrus.exceptions
import arrus.devices.us4r
import arrus.medium
import arrus.metadata
import arrus.params
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.ops.us4r
import arrus.ops.imaging


class AbstractSession(abc.ABC):
    """
    An abstract class of session.

    This class is not intended to be instantiated.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_device(self, id: str):
        """
        Returns a device located at given path.

        :param id: a path to a device, for example '/Us4R:0'
        :return: a device located in a given path.
        """
        raise ValueError("Tried to access an abstract method.")

    @abc.abstractmethod
    def set_current_medium(self, medium: arrus.medium.Medium):
        """
        Sets a medium in the current session context.

        :param medium: medium description to set
        """
        raise ValueError("Tried to access an abstract method.")


@dataclasses.dataclass(frozen=True)
class SessionContext:
    medium: arrus.medium.Medium


class Session(AbstractSession):
    """
    A communication session with the ultrasound system.
    Currently, only localhost session is available.
    """

    def __init__(self, cfg_path: str = None,
                 medium: arrus.medium.Medium = None,
                 mock: dict = None):
        """
        Session constructor.

        :param cfg_path: a path to configuration file
        :param mock: a map device id -> a file that should be used to
          mock the device
        :param medium: medium description to set in context
        """
        super().__init__()
        if not (bool(cfg_path is None) ^ bool(mock is None)):
            raise ValueError("Exactly one of the following parameters should "
                             "be provided: cfg_path, mock.")
        self._session_handle = arrus.core.createSessionSharedHandle(cfg_path)
        self._py_devices = self._create_py_devices(mock)
        self._context = SessionContext(medium=medium)

    def get_device(self, path: str):
        """
        Returns a device identified by a given id.

        The available devices are determined by the initial session settings.

        The handle to device is invalid after the session is closed
        (i.e. the session object is disposed).

        :param path: a path to the device
        :return: a handle to device
        """
        # First, try getting initially available devices.
        if path in self._py_devices:
            return self._py_devices[path]
        # Then try finding a device using arrus core implementation.
        device_handle = self._session_handle.getDevice(path)
        # Cast device to its type class.
        device_id = device_handle.getDeviceId()
        device_type = device_id.getDeviceType()
        specific_device_cast = {
            # Currently only us4r is supported.
            arrus.core.DeviceType_Us4R:
                lambda handle: arrus.devices.us4r.Us4R(
                    arrus.core.castToUs4r(handle),
                    self)

        }.get(device_type, None)
        if specific_device_cast is None:
            raise arrus.exceptions.DeviceNotFoundError(path)
        specific_device = specific_device_cast(device_handle)
        # TODO(pjarosik) key should be an id, not the whole path
        self._py_devices[path] = specific_device
        return specific_device

    def get_session_context(self):
        return self._context

    def set_current_medium(self, medium: arrus.medium.Medium):
        # TODO mutex, forbid when context is frozen (e.g. when us4r is running)
        raise RuntimeError("NYI")

    def _create_py_devices(self, mock):
        # Create mock devices

        devices = {}
        for k, v in mock.items():
            devices["/" + k] = MockMatlab045.load_device(v)

        # Create CPU and GPU devices
        devices["/CPU:0"] = arrus.devices.cpu.CPU(0)
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is not None:
            import cupy
            cupy.cuda.device.Device(0).use()
            devices["/GPU:0"] = arrus.devices.gpu.GPU(0)
        return devices


# ------------------------------------------ MATLAB 0.4.5 LEGACY MOCK DATA
class MockMatlab045:

    @staticmethod
    def load_device(cfg):
        metadata = MockMatlab045._read_metadata(cfg)
        data = cfg["rf"]
        return arrus.devices.mock_us4r.MockUs4R(data, metadata, 0)

    @staticmethod
    def _get_scalar(obj, key):
        return obj[key][0][0]

    @staticmethod
    def _get_vector(obj, key):
        return np.array(obj[key]).flatten()

    @staticmethod
    def _read_metadata(data):
        sys = data["sys"]
        seq = data["seq"]

        # Device
        pitch = MockMatlab045._get_scalar(sys, "pitch")
        curv_radius = - MockMatlab045._get_scalar(sys, "curvRadius")

        probe_model = arrus.devices.probe.ProbeModel(
            model_id=arrus.devices.probe.ProbeModelId(
                manufacturer="nanoecho", name="magprobe"),
            n_elements=MockMatlab045._get_scalar(sys, "nElem"),
            pitch=pitch,
            curvature_radius=curv_radius)
        probe = arrus.devices.probe.ProbeDTO(model=probe_model)
        us4r = arrus.devices.us4r.Us4RDTO(probe=probe, sampling_frequency=65e6)

        # Sequence
        tx_freq = MockMatlab045._get_scalar(seq, "txFreq")
        n_periods = MockMatlab045._get_scalar(seq, "txNPer")
        inverse = MockMatlab045._get_scalar(seq, "txInvert").astype(np.bool)

        pulse = arrus.ops.us4r.Pulse(
            center_frequency=tx_freq, n_periods=n_periods,
            inverse=inverse)

        # In matlab API the element numbering starts from 1
        tx_ap_center_element = MockMatlab045._get_vector(seq, "txCentElem") - 1
        tx_ap_size = MockMatlab045._get_scalar(seq, "txApSize")
        tx_angle = MockMatlab045._get_scalar(seq, "txAng")
        tx_focus = MockMatlab045._get_scalar(seq, "txFoc")
        tx_ap_cent_ang = MockMatlab045._get_vector(seq, "txApCentAng")

        rx_ap_center_element = MockMatlab045._get_vector(seq, "rxCentElem") - 1
        rx_ap_size = MockMatlab045._get_scalar(seq, "rxApSize")
        rx_samp_freq = MockMatlab045._get_scalar(seq, "rxSampFreq")
        pri = MockMatlab045._get_scalar(seq, "txPri")
        fsDivider = MockMatlab045._get_scalar(seq, "fsDivider")
        start_sample = MockMatlab045._get_scalar(seq, "startSample") -1
        end_sample = MockMatlab045._get_scalar(seq, "nSamp")
        tgc_start = MockMatlab045._get_scalar(seq, "tgcStart")
        tgc_slope = MockMatlab045._get_scalar(seq, "tgcSlope")

        sequence = arrus.ops.imaging.LinSequence(
            tx_aperture_center_element=tx_ap_center_element,
            tx_aperture_size=tx_ap_size,
            tx_focus=tx_focus,
            pulse=pulse,
            rx_aperture_center_element=rx_ap_center_element,
            rx_aperture_size=rx_ap_size,
            downsampling_factor=fsDivider,
            rx_sample_range=(start_sample, end_sample),
            pri=pri,
            tgc_start=tgc_start,
            tgc_slope=tgc_slope)

        # Medium
        c = MockMatlab045._get_scalar(seq, "c")
        medium = arrus.medium.MediumDTO("dansk_phantom_1525_us4us",
                                        speed_of_sound=c)

        custom_data = dict()
        custom_data["start_sample"] = MockMatlab045._get_scalar(seq, "startSample") - 1
        custom_data["tx_delay_center"] = MockMatlab045._get_scalar(seq, "txDelCent")
        custom_data["rx_aperture_origin"] = MockMatlab045._get_vector(seq, "rxApOrig")
        custom_data["tx_aperture_center_angle"] = MockMatlab045._get_vector(seq,
                                                                   "txApCentAng")

        # Data characteristic:
        data_char = arrus.metadata.EchoDataDescription(
            sampling_frequency=rx_samp_freq)

        # create context
        context = arrus.metadata.FrameAcquisitionContext(
            device=us4r, sequence=sequence, medium=medium,
            raw_sequence=None,
            custom_data=custom_data)
        return arrus.metadata.Metadata(
            context=context, data_desc=data_char, custom={})
