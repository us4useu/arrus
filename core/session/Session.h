#ifndef ARRUS_CORE_SESSION_SESSION_H
#define ARRUS_CORE_SESSION_SESSION_H

#include <memory>
#include <unordered_map>

#include "core/devices/common.h"
#include "core/devices/Device.h"
#include "core/session/SessionSettings.h"

namespace arrus {

	class Session {
	public:

	    /**
	     * Creates a new session with the provided configuration.
	     *
	     * @param sessionSettings session settings to set.
	     */
		Session(const SessionSettings& sessionSettings);

		/**
		 * Releases all allocated resources and devices.
		 */
		~Session();

		DeviceHandle getDevice(const std::string& deviceId);
		DeviceHandle getDevice(const DeviceId& deviceId);

	private:
//	    std::unordered_map<DeviceId, DeviceHandle> devices;
	};
}


#endif //ARRUS_CORE_SESSION_SESSION_H
