#ifndef ARRUS_CORE_DEVICES_DEVICESETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_DEVICESETTINGSVALIDATOR_H

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/common/validation.h"

namespace arrus::devices {

template<typename D>
class SettingsValidator : public Validator<D> {
public:
    explicit SettingsValidator(const DeviceId &id)
    : Validator<D>(id.toString() + " settings") {}
};

}

#endif //ARRUS_CORE_DEVICES_DEVICESETTINGSVALIDATOR_H
