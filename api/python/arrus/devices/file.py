import arrus.devices.probe
from arrus.devices.device import Device, DeviceId, DeviceType
from arrus.devices.ultrasound import Ultrasound, UltrasoundDTO
import arrus.metadata

DEVICE_TYPE = DeviceType("File")


class File(Device, Ultrasound):
    """
    A handle to ultrasound File device.

    The "file device" represents a device that produces data
    from the provided input file.
    """

    def __init__(self, handle):
        super().__init__()
        self._handle = handle
        self._device_id = DeviceId(
            device_type=DEVICE_TYPE,
            ordinal=self._handle.getDeviceId().getOrdinal()
        )

    def get_device_id(self):
        return self._device_id

    def get_probe_model(self):
        """
        Returns probe model description.
        """
        import arrus.utils.core
        return arrus.utils.core.convert_to_py_probe_model(
            core_model=self._handle.getProbe(0).getModel())

    def set_kernel_context(self, context):
        self._kernel_context = context

    @property
    def current_sampling_frequency(self):
        return self._handle.getCurrentSamplingFrequency()

    @property
    def sampling_frequency(self):
        return self._handle.getSamplingFrequency()

    def get_dto(self):
        import arrus.utils.core
        probe_model = self.get_probe_model()
        probe_dto = arrus.devices.probe.ProbeDTO(model=probe_model)
        return UltrasoundDTO(
            probe=probe_dto,
            sampling_frequency=self.sampling_frequency
        )

    def get_data_description(self, upload_result, sequence, array_id):
        return arrus.metadata.EchoDataDescription(
            sampling_frequency=self.current_sampling_frequency,
            custom={}
        )
