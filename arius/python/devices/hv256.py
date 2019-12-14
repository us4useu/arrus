import arius.python.devices.arius as _arius
import arius.python.devices.device as _device
import arius.python.devices.ihv256 as _hv256
from logging import DEBUG, INFO


class HV256(_device.Device):
    _DEVICE_NAME = "HV256"

    @staticmethod
    def get_card_id(index):
        return _device.Device.get_device_id(_arius.AriusCard._DEVICE_NAME, index)

    def __init__(self, hv256_handle: _hv256.IHV256):
        """
        HV 256 Device. Provides means to steer the voltage
        set on the master Arius card.

        :param card_handle: a handle to the HV256 C++ class.
        """
        super().__init__(HV256._DEVICE_NAME, index=None)
        self.hv256_handle = hv256_handle

    def enable_hv(self):
        self.log(INFO, "Enabling HV.")
        self.hv256_handle.EnableHV()

    def disable_hv(self):
        self.log(INFO, "Disabling HV.")
        self.hv256_handle.DisableHV()

    def set_hv_voltage(self, voltage):
        self.log(INFO, "Setting HV voltage %d" % voltage)
        self.hv256_handle.SetHVVoltage(voltage=voltage)
