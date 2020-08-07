#ifndef ARRUS_CORE_DEVICES_DEVICESETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_DEVICESETTINGSVALIDATOR_H

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/common/validation.h"

namespace arrus {

template<typename D>
class DeviceSettingsValidator : public Validator<D> {
public:
    explicit DeviceSettingsValidator(const DeviceId &id)
    : Validator<D>(id.toString()) {}
};

}

#endif //ARRUS_CORE_DEVICES_DEVICESETTINGSVALIDATOR_H
