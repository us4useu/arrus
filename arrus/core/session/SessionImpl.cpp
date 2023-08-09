#include "arrus/core/session/SessionImpl.h"

#include <gsl/gsl>
#include <memory>

#include <boost/algorithm/string.hpp>

#include "arrus/common/compiler.h"
#include "arrus/common/format.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/devices/utils.h"

#include "arrus/core/api/io/settings.h"
#include "arrus/core/devices/probe/ProbeFactoryImpl.h"
#include "arrus/core/devices/us4r/Us4RFactoryImpl.h"
#include "arrus/core/devices/us4r/Us4RSettingsConverterImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactoryImpl.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMInitializerImpl.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplierFactoryImpl.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterFactoryImpl.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactoryImpl.h"
#include "arrus/core/devices/file/FileFactoryImpl.h"
#include "arrus/core/session/SessionSettings.h"

namespace arrus::session {

using namespace arrus::devices;

#define ASSERT_STATE(expectedState)                                                                                    \
    do {                                                                                                               \
        if (this->state != expectedState) {                                                                            \
            throw ::arrus::IllegalStateException(                                                                      \
                ::arrus::format("Invalid session state, should be: {}", toString(expectedState)));                     \
        }                                                                                                              \
    } while (0)

#define ASSERT_STATE_NOT(excludedState)                                                                                \
    do {                                                                                                               \
        if (this->state == excludedState) {                                                                            \
            throw ::arrus::IllegalStateException(                                                                      \
                ::arrus::format("Invalid session state, should not be: {}", toString(excludedState)));                 \
        }                                                                                                              \
    } while (0)

Session::Handle createSession(const SessionSettings &sessionSettings) {
    return std::make_unique<SessionImpl>(
        sessionSettings,
        std::make_unique<Us4RFactoryImpl>(
            std::make_unique<Us4OEMFactoryImpl>(), std::make_unique<ProbeAdapterFactoryImpl>(),
            std::make_unique<ProbeFactoryImpl>(), std::make_unique<IUs4OEMFactoryImpl>(),
            std::make_unique<IUs4OEMInitializerImpl>(), std::make_unique<Us4RSettingsConverterImpl>(),
            std::make_unique<HighVoltageSupplierFactoryImpl>()),
        std::make_unique<FileFactoryImpl>()
        );
}

Session::Handle createSession(const std::string &filepath) {
    auto settings = arrus::io::readSessionSettings(filepath);
    return createSession(settings);
}

SessionImpl::SessionImpl(
    const SessionSettings &sessionSettings,
    Us4RFactory::Handle us4RFactory,
    FileFactory::Handle fileFactory
    )
    : us4rFactory(std::move(us4RFactory)), fileFactory(std::move(fileFactory)) {
    getDefaultLogger()->log(LogSeverity::DEBUG,
                            arrus::format("Configuring session: {}", ::arrus::toString(sessionSettings)));
    configureDevices(sessionSettings);
}

arrus::devices::Device::RawHandle SessionImpl::getDevice(const std::string &path) {
    // sanitize
    std::string sanitizedPath{path};
    boost::algorithm::trim(sanitizedPath);

    // parse path
    auto [root, tail] = ::arrus::devices::getPathRoot(sanitizedPath);

    auto deviceId = DeviceId::parse(root);
    arrus::devices::Device::RawHandle rootDevice = getDevice(deviceId);

    if (tail.empty()) {
        return rootDevice;
    } else {
        if (isInstanceOf<DeviceWithComponents>(rootDevice)) {
            return ((DeviceWithComponents *) rootDevice)->getDevice(tail);
        } else {
            throw IllegalArgumentException(
                arrus::format("Invalid path '{}', top-level devices can be accessed only.", path));
        }
    }
}

arrus::devices::Device::RawHandle SessionImpl::getDevice(const DeviceId &deviceId) {
    try {
        if(containsKey(devices, deviceId)) {
            return devices.at(deviceId).get();
        }
        else {
            return aliases.at(deviceId);
        }
    } catch (const std::out_of_range &) {
        throw IllegalArgumentException(arrus::format("Device unavailable: {}", deviceId.toString()));
    }
}

void SessionImpl::configureDevices(const SessionSettings &sessionSettings) {
    // Ultrasound systems:
    Ordinal ultrasoundOrdinal = 0;
    // - Us4R:
    for(size_t i = 0; i < sessionSettings.getNumberOfUs4Rs(); ++i) {
        const Us4RSettings &settings = sessionSettings.getUs4RSettings(i);
        Us4R::Handle us4r = us4rFactory->getUs4R(i, settings);
        aliases.emplace(DeviceId(DeviceType::Ultrasound, ultrasoundOrdinal), us4r.get());
        devices.emplace(us4r->getDeviceId(), std::move(us4r));
        ultrasoundOrdinal++;
    }
    // - Files:
    for(size_t i = 0; i < sessionSettings.getNumberOfFiles(); ++i) {
        const FileSettings &settings = sessionSettings.getFileSettings(i);
        File::Handle file = fileFactory->getFile(i, settings);
        aliases.emplace(DeviceId(DeviceType::Ultrasound, ultrasoundOrdinal), file.get());
        devices.emplace(file->getDeviceId(), std::move(file));
        ultrasoundOrdinal++;
    }
}

SessionImpl::~SessionImpl() {
    try {
        this->close();
    } catch(const std::exception &e) {
        getDefaultLogger()->log(LogSeverity::ERROR, arrus::format("Error while closing session: {}", e.what()));
    } catch(...) {
        getDefaultLogger()->log(LogSeverity::ERROR, "Unknown error on session close.");
    }

}

UploadResult SessionImpl::upload(const ops::us4r::Scheme &scheme) {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    ASSERT_STATE(State::STOPPED);

    auto ultrasound = (::arrus::devices::Ultrasound *) getDevice(DeviceId(DeviceType::Ultrasound, 0));
    auto[buffer, metadata] = ultrasound->upload(scheme);
    currentScheme = scheme;
    return UploadResult(buffer, metadata);
}

void SessionImpl::startScheme() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    ASSERT_STATE(State::STOPPED);
    auto ultrasound = (::arrus::devices::Ultrasound *) getDevice(DeviceId(DeviceType::Ultrasound, 0));
    ultrasound->start();
    state = State::STARTED;
}

void SessionImpl::stopScheme() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    auto ultrasound = (::arrus::devices::Ultrasound *) getDevice(DeviceId(DeviceType::Ultrasound, 0));
    ultrasound->stop();
    state = State::STOPPED;
    getDefaultLogger()->log(LogSeverity::INFO, "Scheme stopped.");
}

void SessionImpl::run() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    ASSERT_STATE_NOT(State::CLOSED);

    if (!currentScheme.has_value()) {
        throw IllegalStateException("Upload scheme before running.");
    }
    if (state == State::STOPPED) {
        startScheme();
    } else {
        if (currentScheme.value().getWorkMode() == ops::us4r::Scheme::WorkMode::MANUAL) {
            auto ultrasound = (::arrus::devices::Ultrasound *) getDevice(DeviceId(DeviceType::Ultrasound, 0));
            ultrasound->trigger();
        } else {
            throw IllegalStateException("Scheme already started.");
        }
    }
}

void SessionImpl::close() {
    std::lock_guard<std::recursive_mutex> guard(stateMutex);
    if (this->state == State::CLOSED) {
        getDefaultLogger()->log(LogSeverity::INFO, arrus::format("Session already closed."));
        return;
    }
    if (this->state == State::STARTED) {
        stopScheme();
    }
    getDefaultLogger()->log(LogSeverity::INFO, arrus::format("Closing session."));
    this->devices.clear();
    this->state = State::CLOSED;
}

}// namespace arrus::session
