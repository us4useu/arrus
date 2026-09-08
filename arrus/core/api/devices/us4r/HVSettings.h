#ifndef ARRUS_CORE_API_DEVICES_US4R_HVSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_HVSETTINGS_H

#include <utility>

#include "arrus/core/api/devices/us4r/HVModelId.h"

namespace arrus::devices {

class HVSettings {
public:
    explicit HVSettings(HVModelId modelId, uint8_t voltagePrecisionFactor = 0)
    : modelId(std::move(modelId)), hvVoltagePrecisionFactor(voltagePrecisionFactor) {}

    const HVModelId &getModelId() const {
        return modelId;
    }

    const uint8_t getVoltagePrecisionFactor() const {
        return hvVoltagePrecisionFactor;
    }

private:
    HVModelId modelId;
    uint8_t hvVoltagePrecisionFactor{1};
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_HVSETTINGS_H
