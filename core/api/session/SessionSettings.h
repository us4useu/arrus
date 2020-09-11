#ifndef ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H

#include <utility>
#include <ostream>

#include "arrus/core/api/devices/us4r/Us4RSettings.h"

namespace arrus::session {
class SessionSettings {
public:
    explicit SessionSettings(arrus::devices::Us4RSettings us4RSettings) :
        us4RSettings(std::move(us4RSettings)) {}

    [[nodiscard]] const arrus::devices::Us4RSettings &getUs4RSettings() const {
        return us4RSettings;
    }

private:
    arrus::devices::Us4RSettings us4RSettings;
};
}

#endif //ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
