#ifndef ARRUS_CORE_DEVICES_US4R_VALIDATORS_WATCHDOGSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_VALIDATORS_WATCHDOGSETTINGSVALIDATOR_H

#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/ActiveTerminationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/common/validation.h"

namespace arrus::devices {

class WatchdogSettingsValidator : public Validator<WatchdogSettings> {
public:
    WatchdogSettingsValidator() : Validator("") {}

    void validate(const WatchdogSettings &settings) override {
        if(settings.isEnabled()) {
            float minimumThreshold = 1e-3f;
            float maximumThreshold = 4.0f;
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(settings.getOEMThreshold0(), minimumThreshold, maximumThreshold);
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(settings.getOEMThreshold1(), minimumThreshold, maximumThreshold);
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(settings.getHostThreshold(), minimumThreshold, maximumThreshold);
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_VALIDATORS_WATCHDOGSETTINGSVALIDATOR_H
