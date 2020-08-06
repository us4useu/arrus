#ifndef ARRUS_CORE_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_SESSION_SESSIONSETTINGS_H

#include <utility>

#include "arrus/core/api/devices/us4r/Us4RSettings.h"

namespace arrus {
class SessionSettings {
public:
    SessionSettings(Us4RSettings us4RSettings) :
            us4RSettings(std::move(us4RSettings)) {}

    const Us4RSettings &getUs4RSettings() const {
        return us4RSettings;
    }

private:
    Us4RSettings us4RSettings;
    // TODO log level, output logging, etc.
};
}

#endif //ARRUS_CORE_SESSION_SESSIONSETTINGS_H
