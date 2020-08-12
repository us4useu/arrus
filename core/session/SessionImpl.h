#ifndef ARRUS_CORE_SESSION_SESSIONIMPL_H
#define ARRUS_CORE_SESSION_SESSIONIMPL_H

#include <unordered_map>

#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/devices/DeviceIdHasher.h"

namespace arrus {

class SessionImpl : public Session {
public:
    explicit SessionImpl(const SessionSettings &sessionSettings);

    Device::Handle &getDevice(const std::string &deviceId) override;

    Device::Handle &getDevice(const DeviceId &deviceId) override;

    SessionImpl(SessionImpl const &) = delete;

    void operator=(SessionImpl const &) = delete;

    SessionImpl(SessionImpl const &&) = delete;

    void operator=(SessionImpl const &&) = delete;

private:
    using DeviceMap = std::unordered_map<DeviceId, Device::Handle,
            GET_HASHER_NAME(DeviceId)>;

    DeviceMap configureDevices(const SessionSettings &sessionSettings);

    DeviceMap devices;
};
}

#endif //ARRUS_CORE_SESSION_SESSIONIMPL_H
