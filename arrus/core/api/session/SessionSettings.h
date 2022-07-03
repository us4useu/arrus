#ifndef ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H

#include <ostream>
#include <utility>

#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/api/devices/ultrasound/UltrasoundSettings.h"


namespace arrus::session {

/**
 * Session settings.
 *
 * @see SessionSettingsBuilder
 */
class SessionSettings {
    class Impl;
public:
    /**
     * Creates session to communicate with given Us4R system.
     *
     * @deprecated please use the SessionSettingsBuilder, and create settings for Ultrasound device.
     */
    explicit SessionSettings(arrus::devices::Us4RSettings us4RSettings) {
        this->us4Rs.push_back(std::move(us4RSettings));
    }

    explicit SessionSettings(arrus::devices::UltrasoundSettings settings) {
        this->ultrasounds.push_back(std::move(settings));
    }

    const arrus::devices::Us4RSettings &getUs4RSettings(::arrus::devices::Ordinal id) const {return us4Rs.at(id);}

    /**
     * Returns settings of the first us4R device.
     *
     * @deprecated please use the getUs4RSettings(Ordinal i)
     * @return reference to us4R settings
     */
    const arrus::devices::Us4RSettings &getUs4RSettings() const { return getUs4RSettings(0); }

    /**
     * TODO deprecated
     * @deprecated use getUltrasounds
     */
    const std::vector<arrus::devices::Us4RSettings> &getUs4Rs() const { return us4Rs; }

    const std::vector<arrus::devices::UltrasoundSettings> &getUltrasounds() const { return ultrasounds; }

private:
    std::vector<arrus::devices::Us4RSettings> us4Rs; // TODO: deprecated: convert Us4RSettings to UltrasoundSettings with new Us4RBackendSettings
    std::vector<arrus::devices::UltrasoundSettings> ultrasounds;
};

class SessionSettingsBuilder {
    // TODO
};

}// namespace arrus::session

#endif//ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
