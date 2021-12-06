#ifndef ARRUS_CORE_SESSION_SESSIONIMPL_H
#define ARRUS_CORE_SESSION_SESSIONIMPL_H

#include <unordered_map>
#include <mutex>

#include <arrus/core/devices/us4r/Us4RFactory.h>
#include "arrus/core/api/session/Session.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/devices/DeviceId.h"
#include "arrus/common/utils.h"

namespace arrus::session {

class SessionImpl : public Session {
public:
    SessionImpl(
        const SessionSettings &sessionSettings,
        arrus::devices::Us4RFactory::Handle us4RFactory);

    virtual ~SessionImpl();

    arrus::devices::Device::RawHandle
    getDevice(const std::string &deviceId) override;

    arrus::devices::Device::RawHandle
    getDevice(const arrus::devices::DeviceId &deviceId) override;

    UploadResult upload(const ops::us4r::Scheme &scheme) override;

    void startScheme() override;

    void stopScheme() override;

    void run() override;

    SessionImpl(SessionImpl const &) = delete;

    void operator=(SessionImpl const &) = delete;

    SessionImpl(SessionImpl const &&) = delete;

    void operator=(SessionImpl const &&) = delete;
    void close() override;

private:
    ARRUS_DEFINE_ENUM_WITH_TO_STRING(
            State,
            // The connection with devices is established, but the input sources do not produce
            // any new data right now.
            (STOPPED)
            // All the input sources are producing new data.
            (STARTED)
            // The connection with devices is closed.
            (CLOSED)
    );

    using DeviceMap = std::unordered_map<arrus::devices::DeviceId, arrus::devices::Device::Handle,
                                         GET_HASHER_NAME(arrus::devices::DeviceId)>;

    DeviceMap configureDevices(const SessionSettings &sessionSettings);


    DeviceMap devices;
    arrus::devices::Us4RFactory::Handle us4rFactory;
    std::recursive_mutex stateMutex;
    std::optional<ops::us4r::Scheme> currentScheme;
    State state{State::STOPPED};
};


}

#endif //ARRUS_CORE_SESSION_SESSIONIMPL_H
