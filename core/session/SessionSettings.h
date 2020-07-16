#ifndef ARRUS_CORE_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_SESSION_SESSIONSETTINGS_H

#include <vector>

#include "core/devices/common.h"
#include "core/devices/us4oem/Us4OEMSettings.h"

namespace arrus {

/**
 * Session settings.
 *
 * This class is meant to be a part of ARRUS API.
 */
class SessionSettings {
public:
	SessionSettings(const std::vector<Us4OEMSettings>& us4oemSettings)
		: us4oemSettings(us4oemSettings)
	{}

	const Us4OEMSettings &getUs4OEMSettings(const Ordinal ordinal) const {
		return this->us4oemSettings[ordinal];
	}

private:
	std::vector<Us4OEMSettings> us4oemSettings;
};

}

#endif //ARRUS_CORE_SESSION_SESSIONSETTINGS_H
