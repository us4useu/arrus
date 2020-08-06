#ifndef ARRUS_CORE_DEVICES_US4R_US4RSETTINGS_H
#define ARRUS_CORE_DEVICES_US4R_US4RSETTINGS_H

#include <utility>
#include <map>

#include "arrus/core/devices/us4oem/Us4OEMSettings.h"
#include "arrus/core/devices/DeviceId.h"

namespace arrus {

class Us4RSettings {
public:

    /**
     * The position of the us4OEM settings in the vector is the Us4OEM
     * device ordinal.
     */
    using Us4OEMSettingsCollection = std::vector<Us4OEMSettings>;

    explicit Us4RSettings(Us4OEMSettingsCollection us4oemSettings)
    : us4oemSettings(std::move(us4oemSettings)) {}

    [[nodiscard]] const Us4OEMSettingsCollection& getUs4oemSettings() const {
        return this->us4oemSettings;
    }

private:
    Us4OEMSettingsCollection us4oemSettings;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RSETTINGS_H
