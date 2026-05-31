#ifndef ARRUS_CORE_SESSION_SESSIONIMPL_H
#define ARRUS_CORE_SESSION_SESSIONIMPL_H

#include <unordered_map>
#include <mutex>

#include "arrus/core/devices/us4r/Us4RFactory.h"
#include "arrus/core/devices/file/FileFactory.h"
#include "arrus/core/devices/gpu/GpuFactory.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/devices/DeviceId.h"
#include "arrus/common/utils.h"

namespace arrus::session {

class SessionImpl : public Session {
public:
    SessionImpl(
        const SessionSettings &sessionSettings,
        arrus::devices::Us4RFactory::Handle us4RFactory,
        arrus::devices::FileFactory::Handle fileFactory,
        arrus::devices::GpuFactory::Handle gpuFactory
        );
    ~SessionImpl() override;
    using Session::getDevice;  // unhide the std::string shim from the base
    arrus::devices::Device::RawHandle
    getDeviceNative(const std4us::String &deviceId) override;
    arrus::devices::Device::RawHandle
    getDevice(const arrus::devices::DeviceId &deviceId) override;
    UploadResult upload(const ops::us4r::Scheme &scheme) override;
    void startScheme() override;
    void stopScheme() override;
    void runNative(bool async, std4us::Optional<long long> timeout) override;
    SessionImpl(SessionImpl const &) = delete;
    void operator=(SessionImpl const &) = delete;
    SessionImpl(SessionImpl const &&) = delete;
    void operator=(SessionImpl const &&) = delete;
    void close() override;
    void setParameters(const Parameters &params) override;
    State getCurrentState() override;
    UploadResult setSubsequenceNative(uint16 start, uint16 end, std4us::Optional<float> sri, uint16 arrayId) override;
    UploadResult setSubsequences(const std::vector<Slice> &slices, const std::vector<std::optional<float>> &sris) override;
    using Session::hasDevice;  // unhide the std::string shim from the base
    bool hasDeviceNative(const std4us::String &deviceId) const override;
    bool hasDevice(const devices::DeviceId &deviceId) const override;

private:
    ARRUS_DEFINE_ENUM_TO_STRING(
            State,
            // The connection with devices is established, but the input sources do not produce
            // any new data right now.
            (STOPPED)
            // All the input sources are producing new data.
            (STARTED)
            // The connection with devices is closed.
            (CLOSED)
    );
    void verifyScheme(const ops::us4r::Scheme &scheme);
    devices::DeviceId parseDeviceId(const std::string &path, std::string &tail, devices::DeviceId &deviceId) const;

    using DeviceMap = std::unordered_map<
        arrus::devices::DeviceId,
        arrus::devices::Device::Handle,
        GET_HASHER_NAME(arrus::devices::DeviceId)>;

    using AliasMap = std::unordered_map<
        arrus::devices::DeviceId,
        arrus::devices::Device::RawHandle,
        GET_HASHER_NAME(arrus::devices::DeviceId)>;

    void configureDevices(const SessionSettings &sessionSettings);

    DeviceMap devices;
    AliasMap aliases;
    arrus::devices::Us4RFactory::Handle us4rFactory;
    arrus::devices::FileFactory::Handle fileFactory;
    arrus::devices::GpuFactory::Handle gpuFactory;
    std::recursive_mutex stateMutex;
    std::optional<ops::us4r::Scheme> currentScheme;
    State state{State::STOPPED};
    std::string sanitizeDeviceId(const std::string &path) const;
};


}

#endif //ARRUS_CORE_SESSION_SESSIONIMPL_H
