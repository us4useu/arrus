#ifndef ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H

#include <ostream>
#include <utility>

#include "arrus/core/api/devices/us4r/Us4RSettings.h"

namespace arrus::session {

class SessionSettings {
public:
    /**
     * Creates session to communicate with given Us4R system.
     *
     * @deprecated please use the SessionSettingsBuilder, theb create settings for an UltrasoundScanner with the
     * proper backend.
     */
    explicit SessionSettings(arrus::devices::Us4RSettings us4RSettings) {
        this->us4Rs.push_back(std::move(us4RSettings));
    }

    const arrus::devices::Us4RSettings &getUs4RSettings(::arrus::devices::Ordinal id) const {return us4Rs.at(id);}

    /**
     * Returns settings of the first us4R device in the system.
     *
     * @deprecated please use the getUs4RSettings(Ordinal i)
     * @return reference to us4R settings
     */
    const arrus::devices::Us4RSettings &getUs4RSettings() const { return getUs4RSettings(0); }

private:
    friend class SessionSettingsBuilder;
    SessionSettings() {}
    std::vector<arrus::devices::Us4RSettings> us4Rs;
};

class SessionSettingsBuilder {
public:
    SessionSettingsBuilder() {

    }

    SessionSettingsBuilder &setDictionary() {

    }

    SessionSettingsBuilder &addUltrasoundScanner() {

    }
private:
    SessionSettings current;
};

}// namespace arrus::session

#endif//ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
