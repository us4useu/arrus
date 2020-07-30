#ifndef ARRUS_CORE_SESSION_SESSION_H
#define ARRUS_CORE_SESSION_SESSION_H

#include <memory>
#include <unordered_map>

#include "core/devices/Device.h"
#include "core/devices/DeviceId.h"
#include "core/session/SessionSettings.h"
#include "core/common/exceptions.h"

namespace arrus {

class Session {
public:
    /**
     * Creates a new session with the provided configuration.
     *
     * @param sessionSettings session settings to set.
     */
    explicit Session(const SessionSettings &sessionSettings);

    /**
     * Releases all allocated resources and devices.
     */
    ~Session() = default;

    Session(const Session &) = delete;

    Session(const Session &&) = delete;

    Session &operator=(const Session &) = delete;

    Session &operator=(const Session &&) = delete;

    /**
     * Returns a handle to device with given Id.
     *
     * @param deviceId device identifier
     * @return a device handle
     */
    Device::Handle& getDevice(const std::string &deviceId);

    /**
     * Returns a handle to device with given Id.
     *
     * @param deviceId device identifier
     * @return a device handle
     */
    Device::Handle& getDevice(const DeviceId &deviceId);

private:
    using DeviceMap = std::unordered_map<DeviceId, Device::Handle,
            GET_HASHER_NAME(DeviceId)>;

    DeviceMap configureDevices(const SystemSettings &settings);

    DeviceMap devices;
};
}


#endif //ARRUS_CORE_SESSION_SESSION_H
