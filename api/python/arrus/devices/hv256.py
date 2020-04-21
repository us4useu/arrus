from logging import INFO, WARN

import arrus.devices.device as _device
import arrus.devices.ihv256 as _hv256

import arrus.devices.arius as _arius


class HV256(_device.Device):
    _DEVICE_NAME = "HV256"
    _N_RETRIES = 2

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
        """
        Enables HV power supplier.
        """
        self.log(INFO, "Enabling HV.")

        for i in range(HV256._N_RETRIES):
            try:
                self.hv256_handle.EnableHV()
                return
            except RuntimeError as e:
                # TODO(pjarosik) write time out should be handled on lower level
                # see Arius-software#29
                self.log(WARN,
                    ("An error occurred while enabling HV: '%s'" % (str(e)))
                    + (". Trying again..." if (i+1) < HV256._N_RETRIES else "")
                )


    def disable_hv(self):
        """
        Disables HV power supplier.
        """
        self.log(INFO, "Disabling HV.")
        self.hv256_handle.DisableHV()

    def set_hv_voltage(self, voltage):
        """
        Sets HV voltage to a given value.

        :param voltage: voltage to set [V]
        """
        self.log(INFO, "Setting HV voltage to %d [V]" % voltage)
        for i in range(HV256._N_RETRIES):
            try:
                self.hv256_handle.SetHVVoltage(voltage=voltage)
                return
            except RuntimeError as e:
                # TODO(pjarosik) write time out should be handled on lower level
                # see Arius-software#29
                self.log(WARN,
                 ("An error occurred while setting HV voltage: '%s'" % (str(e)))
                 + (". Trying again..." if (i+1) < HV256._N_RETRIES else "")
                )

