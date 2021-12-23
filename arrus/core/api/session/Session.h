#ifndef ARRUS_CORE_API_SESSION_SESSION_H
#define ARRUS_CORE_API_SESSION_SESSION_H

#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/common/macros.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/session/SessionSettings.h"
#include "arrus/core/api/session/UploadResult.h"

namespace arrus::session {

/**
 * A communication session with the device.
 */
class Session {
public:
    using Handle = std::unique_ptr<Session>;

    /**
     * Returns a handle to device with given Id. The string format is:
     * /DeviceType:Ordinal, e.g. "/Us4R:0".
     *
     * @param deviceId device identifier
     * @return a handle to the device
     */
    virtual arrus::devices::Device * getDevice(const std::string &deviceId) = 0;

    /**
     * Returns a handle to device with given Id.
     *
     * @param deviceId device identifier
     * @return a handle to the device
     */
    virtual arrus::devices::Device * getDevice(const arrus::devices::DeviceId &deviceId) = 0;

    /**
     * Uploads a given scheme on the available devices.
     *
     * Currently, the scheme upload is performed on the Us4R:0 device only.
     *
     * After uploading a new sequence the previously returned output buffers will be in invalid state.
     *
     * @param scheme scheme to upload
     * @return upload result information
     */
    virtual UploadResult upload(const ::arrus::ops::us4r::Scheme &scheme) = 0;

    /**
     * Starts currently uploaded scheme.
     */
    virtual void startScheme() = 0;

    /**
     * Stops currently uploaded scheme.
     */
    virtual void stopScheme() = 0;

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