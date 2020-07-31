#ifndef ARRUS_CORE_DEVICES_US4R_US4RSETTINGS_H
#define ARRUS_CORE_DEVICES_US4R_US4RSETTINGS_H

#include <utility>
#include <map>

#include "core/devices/us4oem/Us4OEMSettings.h"
#include "core/devices/DeviceId.h"

namespace arrus {

class Us4RSettings {
public:
    using Us4OEMSettingsMap = std::map<Ordinal, Us4OEMSettings>;

    explicit Us4RSettings(const Us4OEMSettingsMap us4oemSettingsMap)
    : us4oemSettingsMap(std::move(us4oemSettingsMap)) {}

    [[nodiscard]] const Us4OEMSettingsMap& getUs4oemSettings() const {
        return this->us4oemSettingsMap;
    }

private:
    Us4OEMSettingsMap us4oemSettingsMap;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RSETTINGS_H
