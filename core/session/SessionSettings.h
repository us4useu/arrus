#ifndef ARRUS_CORE_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_SESSION_SESSIONSETTINGS_H

#include <utility>

#include "core/session/SystemSettings.h"

namespace arrus {
class SessionSettings {
public:
    SessionSettings(SystemSettings systemSettings) :
            systemSettings(std::move(systemSettings)) {}

    const SystemSettings &getSystemSettings() const {
        return systemSettings;
    }

private:
    SystemSettings systemSettings;
    // TODO log level, output logging, etc.
};
}

#endif //ARRUS_CORE_SESSION_SESSIONSETTINGS_H
