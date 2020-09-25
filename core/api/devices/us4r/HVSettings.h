#ifndef ARRUS_CORE_API_DEVICES_US4R_HVSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_HVSETTINGS_H

#include <utility>

#include "arrus/core/api/devices/us4r/HVModelId.h"

namespace arrus::devices {

class HVSettings {
public:
    explicit HVSettings(HVModelId modelId)
    : modelId(std::move(modelId)) {}

    [[nodiscard]] const HVModelId &getModelId() const {
        return modelId;
    }

private:
    HVModelId modelId;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_HVSETTINGS_H
