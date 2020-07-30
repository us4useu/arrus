#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H

#include "core/common/validation.h"
#include "core/devices/us4oem/Us4OEMSettings.h"

namespace arrus {

class Us4OEMSettingsValidator: public Validator<Us4OEMSettings> {

public:
    void validate(const Us4OEMSettings &obj) override {
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
