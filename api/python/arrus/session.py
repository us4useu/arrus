import logging
import abc
import numpy as np

_logger = logging.getLogger(__name__)

import arrus.devices.us4r
import arrus.ops
import arrus.devices.probe
import arrus.medium
import arrus.metadata
import arrus.params

_ARRUS_PATH_ENV = "ARRUS_PATH"


class AbstractSession(abc.ABC):

    def __init__(self, cfg):
        self._devices = self._load_devices(cfg)

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

    def get_devices(self):
        """
        Returns a list of all devices available in this session.

        :return: a list of available devices
        """
        return self._devices

    @abc.abstractmethod
    def run(self, operation: arrus.ops.Operation, feed_dict: dict):
        raise ValueError("This type of session cannot run operations.")

    @abc.abstractmethod
    def _load_devices(self, cfg):
        pass


class MockSession(AbstractSession):

    def __init__(self, dataset):
        super().__init__(dataset)

    def run(self, operation: arrus.ops.Operation, feed_dict: dict):
        raise ValueError("This type of session cannot run operations.")

    def _load_devices(self, cfg):
        metadata = self._read_metadata(cfg)
        data = cfg["rf"].T
        return {
            "Us4R:0": arrus.devices.us4r.MockUs4R(data, metadata, "Us4R", 0)
        }

    def _get_scalar(self, obj, key):
        return obj[key][0][0].flatten()[0]

    def _get_vector(self, obj, key):
        return obj[key][0][0].flatten()

    def _read_metadata(self, data):
        sys = data["sys"]
        seq = data["seq"]

        # Device
        pitch = self._get_scalar(sys, "pitch")
        curv_radius = - self._get_scalar(sys, "curvRadius")

        probe_model = arrus.devices.probe.ProbeModel(
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

        sequence = arrus.ops.LinSequence(
            tx_aperture_center_element=tx_ap_center_element,
            tx_aperture_size=tx_ap_size,
            tx_focus=tx_focus,
            tx_angle=tx_angle,
            pulse=pulse,
            rx_aperture_center_element=rx_ap_center_element,
            rx_aperture_size=rx_ap_size,
            sampling_frequency=rx_samp_freq
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
        data_char = arrus.metadata.EchoSignalDataDescription(
            sampling_frequency=rx_samp_freq)

        # create context
        context = arrus.metadata.FrameAcquisitionContext(
            device=us4r, sequence=sequence, medium=medium,
            custom_data=custom_data)
        return arrus.metadata.Metadata(
            context=context, data_desc=data_char, custom={})
