#ifndef ARRUS_CORE_DEVICES_US4R_VALIDATORS_TGCSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_VALIDATORS_TGCSETTINGSVALIDATOR_H

#include "arrus/core/devices/us4r/TgcSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/common/validation.h"

namespace ::arrus::devices {

class TgcSettingsValidator : public Validator<TgcSettings> {
public:

    void validate(const TgcSettings &settings) override {
        validateDtgcAttenuation(settings.getDtgcAttenuation());
        validatePgaGain(settings.getPgaGain());
        validateLnaGain(settings.getLnaGain());

        // Only one of the following can be used.
        ARRUS_VALIDATOR_EXPECT_TRUE_M(!(settings.getDtgcAttenuation().hasVa && settings.get))

        if(! tgcCurve.empty()) {
            // DTGC

        }

        if(pgaGain != 30 || lnaGain != 24) {

        } else {
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(tgcCurve.size(), size_t(0), size_t(Us4OEM));
            auto[min, max] = getTgcMinMax(pgaGain, lnaGain);
            ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_V(tgcCurve, min, max);
        }
    }

    void validateDtgcAttenuation(uint8 value) {
        expectOneOf("dtgc attenuation", value, DTGCAttenuationValueMap::getInstance().getAvailableValues());
    }

    void validatePgaGain(uint8 value) {
        expectOneOf("pga gain", value, PGAGainValueMap::getInstance().getAvailableValues());
    }

    void validateLnaGain(uint8 value) {
        expectOneOf("lna gain", value, LNAGainValueMap::getInstance().getAvailableValues());
    }

    void
};



}

#endif //ARRUS_CORE_DEVICES_US4R_VALIDATORS_TGCSETTINGSVALIDATOR_H
