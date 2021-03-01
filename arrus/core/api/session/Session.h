#ifndef ARRUS_CORE_API_SESSION_SESSION_H
#define ARRUS_CORE_API_SESSION_SESSION_H

#include "arrus/core/api/common/macros.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::session {

class Session {
public:
    using Handle = std::unique_ptr<Session>;

    /**
     * Returns a handle to device with given Id.
     *
     * @param deviceId device identifier
     * @return a device handle
     */
    virtual arrus::devices::Device *
    getDevice(const std::string &deviceId) = 0;

    /**
     * Returns a handle to device with given Id.
     *
     * @param deviceId device identifier
     * @return a device handle
     */
    virtual arrus::devices::Device *
    getDevice(const arrus::devices::DeviceId &deviceId) = 0;

    virtual void close() = 0;

    virtual ~Session() = default;
};

/**
* Creates a new session with the provided configuration.
*
* @param sessionSettings session settings to set.
* @return a unique handle to session
*/
ARRUS_CPP_EXPORT
Session::Handle createSession(const SessionSettings &sessionSettings);

/**
* Reads given configuration file and returns a handle to new session.
*
* @param filepath a path to session settings
* @return a unique handle to session
*/
ARRUS_CPP_EXPORT
Session::Handle createSession(const std::string& filepath);
}


#endif //ARRUS_CORE_API_SESSION_SESSION_H