#ifndef ARRUS_CORE_SESSION_SYSTEMSETTINGS_H
#define ARRUS_CORE_SESSION_SYSTEMSETTINGS_H

#include <utility>
#include <map>

#include "core/devices/us4oem/Us4OEMSettings.h"

namespace arrus {

/**
 * Session settings.
 *
 * This class is meant to be a part of ARRUS API.
 */
class SystemSettings {
public:
    using Us4OEMSettingsMap = std::map<Ordinal, Us4OEMSettings>;

    explicit SystemSettings(Us4OEMSettingsMap us4oemSettingsMap)
            : us4oemSettingsMap(std::move(us4oemSettingsMap)) {}

    [[nodiscard]] const Us4OEMSettingsMap& getUs4oemSettings() const {
        return this->us4oemSettingsMap;
    }

private:
    Us4OEMSettingsMap us4oemSettingsMap;
};

}

#endif //ARRUS_CORE_SESSION_SYSTEMSETTINGS_H
