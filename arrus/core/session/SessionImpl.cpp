#include "arrus/core/session/SessionImpl.h"

#include <gsl/gsl>
#include <memory>

#include <boost/algorithm/string.hpp>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/format.h"
#include "arrus/common/compiler.h"
#include "arrus/core/devices/utils.h"

#include "arrus/core/devices/us4r/Us4RFactoryImpl.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactoryImpl.h"
#include "arrus/core/devices/probe/ProbeFactoryImpl.h"
#include "arrus/core/devices/us4r/Us4RSettingsConverterImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMInitializerImpl.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterFactoryImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactoryImpl.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplierFactoryImpl.h"
#include "arrus/core/session/SessionSettings.h"
#include "arrus/core/api/io/settings.h"

namespace arrus::session {

using namespace arrus::devices;

#define ASSERT_STATE(expectedState) \
do {                                    \
    if(this->state != expectedState)  { \
                                        \
        throw ::arrus::IllegalStateException(::arrus::format("Invalid session state, should be: {}", \
                                             toString(expectedState))); \
    }                                   \
} while(0)

#define ASSERT_STATE_NOT(excludedState) \
do {                                    \
    if(this->state == excludedState)  { \
                                        \
        throw ::arrus::IllegalStateException(::arrus::format("Invalid session state, should not be: {}", \
                                             toString(excludedState))); \
    }                                   \
} while(0)

Session::Handle createSession(const SessionSettings &sessionSettings) {
    return std::make_unique<SessionImpl>(
        sessionSettings,
        std::make_unique<Us4RFactoryImpl>(
            std::make_unique<Us4OEMFactoryImpl>(),
            std::make_unique<ProbeAdapterFactoryImpl>(),
            std::make_unique<ProbeFactoryImpl>(),
            std::make_unique<IUs4OEMFactoryImpl>(),
            std::make_unique<IUs4OEMInitializerImpl>(),
            std::make_unique<Us4RSettingsConverterImpl>(),
            std::make_unique<HighVoltageSupplierFactoryImpl>()
        )
    );
}

Session::Handle createSession(const std::string& filepath) {
    auto settings = arrus::io::readSessionSettings(filepath);
    return createSession(settings);
}

SessionImpl::SessionImpl(const SessionSettings &sessionSettings, Us4RFactory::Handle us4RFactory)
    : us4rFactory(std::move(us4RFactory)) {
    getDefaultLogger()->log(LogSeverity::DEBUG, format("Configuring session: {}", ::arrus::toString(sessionSettings)));
    devices = configureDevices(sessionSettings);
}

arrus::devices::Device::RawHandle SessionImpl::getDevice(const std::string &path) {
    // sanitize
    std::string sanitizedPath{path};
    boost::algorithm::trim(sanitizedPath);

    // parse path
    auto[root, tail] = ::arrus::devices::getPathRoot(sanitizedPath);

    auto deviceId = DeviceId::parse(root);
    arrus::devices::Device::RawHandle rootDevice = getDevice(deviceId);

    if(tail.empty()) {
        return rootDevice;
    } else {
        if(isInstanceOf<DeviceWithComponents>(rootDevice)) {
            return ((DeviceWithComponents *) rootDevice)->getDevice(tail);
        } else {
            throw IllegalArgumentException(arrus::format(
                "Invalid path '{}', top-level devices can be accessed only.",
                path
            ));
        }
    }
}

arrus::devices::Device::RawHandle SessionImpl::getDevice(const DeviceId &deviceId) {
    try {
        return devices.at(deviceId).get();
    } catch(const std::out_of_range &) {
        throw IllegalArgumentException(
            arrus::format("Device unavailable: {}", deviceId.toString()));
    }
}

SessionImpl::DeviceMap
SessionImpl::configureDevices(const SessionSettings &sessionSettings) {
    DeviceMap result;
    // Configuring Us4R.
    const Us4RSettings &us4RSettings = sessionSettings.getUs4RSettings();
    Us4R::Handle us4r = us4rFactory->getUs4R(0, us4RSettings);
    result.emplace(us4r->getDeviceId(), std::move(us4r));
    return result;
}

SessionImpl::~SessionImpl() {
    this->close();
}

UploadResult SessionImpl::upload(const ops::us4r::Scheme &scheme) {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    ASSERT_STATE(State::STOPPED);

    auto us4r = (::arrus::devices::Us4R *) getDevice(DeviceId(DeviceType::Us4R, 0));

    // Allocate appropriate input buffer.
    auto &outputBufferSpec = scheme.getOutputBuffer();
    auto &placement = outputBufferSpec.getPlacement();
    auto device = getDevice(placement);
    // Get memory allocator
    // allocate - get NdArray
    // Pass the NdArray to the upload function

    auto[buffer, fcm] = us4r->upload(scheme.getTxRxSequence(), scheme.getRxBufferSize(),
                                     scheme.getWorkMode(), outputBufferSpec);

    std::unordered_map<std::string, std::shared_ptr<void>> metadataMap;
    metadataMap.emplace("frameChannelMapping", std::move(fcm));
    auto constMetadata = std::make_shared<UploadConstMetadata>(metadataMap);
    currentScheme = scheme;
    return UploadResult(buffer, constMetadata);
}

void SessionImpl::startScheme() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    ASSERT_STATE(State::STOPPED);
    auto us4r = (::arrus::devices::Us4R *) getDevice(DeviceId(DeviceType::Us4R, 0));
    us4r->start();
    state = State::STARTED;
}

void SessionImpl::stopScheme() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    auto us4r = (::arrus::devices::Us4R *) getDevice(DeviceId(DeviceType::Us4R, 0));
    us4r->stop();
    state = State::STOPPED;
}

void SessionImpl::run() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    ASSERT_STATE_NOT(State::CLOSED);

    if(!currentScheme.has_value()) {
        throw IllegalStateException("Upload scheme before running.");
    }
    if(state == State::STOPPED) {
        startScheme();
    } else {
        if(currentScheme.value().getWorkMode() == ops::us4r::Scheme::WorkMode::MANUAL) {
            auto us4r = (::arrus::devices::Us4RImpl *)getDevice(DeviceId(DeviceType::Us4R, 0));
            us4r->trigger();
        }
        else {
            throw IllegalStateException("Scheme already started.");
        }
    }
}

void SessionImpl::close() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    if(this->state == State::CLOSED) {
        getDefaultLogger()->log(LogSeverity::INFO, arrus::format("Session already closed."));
        return;
    }
    if(this->state == State::STARTED) {
        stopScheme();
    }
    getDefaultLogger()->log(LogSeverity::INFO, arrus::format("Closing session."));
    this->devices.clear();
    this->state = State::CLOSED;
}



}