#ifndef ARRUS_CORE_API_DEVICES_US4R_DIGITALBACKPLANESETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_DIGITALBACKPLANESETTINGS_H

#include <utility>

#include "DigitalBackplaneId.h"

namespace arrus::devices {

class DigitalBackplaneSettings {
public:
    explicit DigitalBackplaneSettings(DigitalBackplaneId modelId) : modelId(std::move(modelId)) {}

    const DigitalBackplaneId &getModelId() const { return modelId; }

private:
    DigitalBackplaneId modelId;
};
}

#endif//ARRUS_CORE_API_DEVICES_US4R_DIGITALBACKPLANESETTINGS_H
