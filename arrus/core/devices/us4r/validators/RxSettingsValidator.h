#ifndef ARRUS_CORE_DEVICES_US4R_VALIDATORS_RXSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_VALIDATORS_RXSETTINGSVALIDATOR_H

#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/ActiveTerminationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/common/validation.h"

namespace arrus::devices {

class RxSettingsValidator : public Validator<RxSettings> {
public:
    RxSettingsValidator() : Validator("rx settings") {}

    void validate(const RxSettings &settings) override {
        ARRUS_VALIDATOR_EXPECT_TRUE_M(
                !settings.getDtgcAttenuation().has_value() || settings.getTgcSamples().empty(),
                "At most one of the following parameters can be specified: dtgc attenuation, analog tgc curve.");
        expectOneOf("pga gain", settings.getPgaGain(), PGAGainValueMap::getInstance().getAvailableValues());
        expectOneOf("lna gain", settings.getLnaGain(), LNAGainValueMap::getInstance().getAvailableValues());
        expectOneOf("lpf cutoff", settings.getLpfCutoff(), LPFCutoffValueMap::getInstance().getAvailableValues());

        // Only one of the following can be used.
        if(settings.getDtgcAttenuation().has_value()) {
            expectOneOf("dtgc attenuation",
                        settings.getDtgcAttenuation().value(), DTGCAttenuationValueMap::getInstance().getAvailableValues());
        }
        if(settings.getActiveTermination().has_value()) {
            expectOneOf("active termination",
                        settings.getActiveTermination().value(), ActiveTerminationValueMap::getInstance().getAvailableValues());
        }
        if(!settings.getTgcSamples().empty()) {
            ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    !settings.isApplyTgcCharacteristic()
                            || (settings.getPgaGain() == 30 and settings.getLnaGain() == 24),
                    "Currently TGC characteristic can be automatically compensated only for "
                    "PGA gain = 30 and LNA gain = 24");
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(settings.getTgcSamples().size(),
                                            size_t(0), size_t(Us4OEMImpl::TGC_N_SAMPLES));
            auto[min, max] = RxSettings::getTgcMinMax(settings.getPgaGain(), settings.getLnaGain());
            ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_V(settings.getTgcSamples(), min, max);
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_VALIDATORS_RXSETTINGSVALIDATOR_H
