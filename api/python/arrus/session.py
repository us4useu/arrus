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

    def __init__(self, cfg_path: str, medium: arrus.medium.Medium = None):
        """
        Session constructor.

        :param cfg_path: a path to configuration file
        :param medium: medium description to set in context
        """
        super().__init__()
        self._session_handle = arrus.core.createSessionSharedHandle(cfg_path)
        self._py_devices = {}
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
        # First, try getting device defined in python language.
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


# ------------------------------------------ LEGACY MOCK
class MockSession(AbstractSession):

    def __init__(self, dataset):
        super().__init__()
        self._devices = self._load_devices(dataset)

    def get_device(self, id: str):
        """
        Returns a device located at given path.

        :param id: a path to a device, for example '/Us4R:0'
        :return: a device located in a given path.
        """
        dev_path = id.split("/")[1:]
        if len(dev_path) != 1:
            raise ValueError(
                "Invalid path, top-level devices can be accessed only.")
        dev_id = dev_path[0]
        return self._devices[dev_id]

    def set_current_medium(self, medium):
        pass

    def _load_devices(self, cfg):
        metadata = self._read_metadata(cfg)
        data = cfg["rf"]
        devices = {
            "Us4R:0": arrus.devices.us4r.MockUs4R(data, metadata, 0),
            "CPU:0": arrus.devices.cpu.CPU(0),
        }
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is not None:
            import cupy
            cupy.cuda.device.Device(0).use()
            devices["GPU:0"] = arrus.devices.gpu.GPU(0)
        return devices

    def _get_scalar(self, obj, key):
        return obj[key][0][0]

    def _get_vector(self, obj, key):
        return np.array(obj[key]).flatten()

    def _read_metadata(self, data):
        sys = data["sys"]
        seq = data["seq"]

        # Device
        pitch = self._get_scalar(sys, "pitch")
        curv_radius = - self._get_scalar(sys, "curvRadius")

        probe_model = arrus.devices.probe.ProbeModel(
            arrus.devices.probe.ProbeModelId(
                manufacturer="nanoecho", name="magprobe"),
            self._get_scalar(sys, "nElem"), pitch, curv_radius)
        probe = arrus.devices.probe.ProbeDTO(model=probe_model)
        us4r = arrus.devices.us4r.Us4RDTO(probe=probe)

        # Sequence
        tx_freq = self._get_scalar(seq, "txFreq")
        n_periods = self._get_scalar(seq, "txNPer")
        inverse = self._get_scalar(seq, "txInvert").astype(np.bool)

        pulse = arrus.params.SineWave(
            center_frequency=tx_freq, n_periods=n_periods,
            inverse=inverse)

        # In matlab API the element numbering starts from 1
        tx_ap_center_element = self._get_vector(seq, "txCentElem") - 1
        tx_ap_size = self._get_scalar(seq, "txApSize")
        tx_angle = self._get_scalar(seq, "txAng")
        tx_focus = self._get_scalar(seq, "txFoc")
        tx_ap_cent_ang = self._get_vector(seq, "txApCentAng")

        rx_ap_center_element = self._get_vector(seq, "rxCentElem") - 1
        rx_ap_size = self._get_scalar(seq, "rxApSize")
        rx_samp_freq = self._get_scalar(seq, "rxSampFreq")
        pri = self._get_scalar(seq, "txPri")

        sequence = arrus.ops.LinSequence(
            tx_aperture_center_element=tx_ap_center_element,
            tx_aperture_size=tx_ap_size,
            tx_focus=tx_focus,
            tx_angle=tx_angle,
            pulse=pulse,
            rx_aperture_center_element=rx_ap_center_element,
            rx_aperture_size=rx_ap_size,
            sampling_frequency=rx_samp_freq,
            pri=pri
        )

        # Medium
        c = self._get_scalar(seq, "c")
        medium = arrus.medium.MediumDTO("dansk_phantom_1525_us4us",
                                        speed_of_sound=c)

        custom_data = dict()
        custom_data["start_sample"] = self._get_scalar(seq, "startSample") - 1
        custom_data["tx_delay_center"] = self._get_scalar(seq, "txDelCent")
        custom_data["rx_aperture_origin"] = self._get_vector(seq, "rxApOrig")
        custom_data["tx_aperture_center_angle"] = self._get_vector(seq,
                                                                   "txApCentAng")

        # Data characteristic:
        data_char = arrus.metadata.EchoDataDescription(
            sampling_frequency=rx_samp_freq)

        # create context
        context = arrus.metadata.FrameAcquisitionContext(
            device=us4r, sequence=sequence, medium=medium,
            custom_data=custom_data)
        return arrus.metadata.Metadata(
            context=context, data_desc=data_char, custom={})
