#include "arrus/core/session/SessionImpl.h"

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

namespace arrus {

Session::Handle createSession(const SessionSettings &sessionSettings) {
    return std::make_unique<SessionImpl>(
            sessionSettings,
            std::make_unique<Us4RFactoryImpl>(
                std::make_unique<Us4OEMFactoryImpl>(),
                std::make_unique<ProbeAdapterFactoryImpl>(),
                std::make_unique<ProbeFactoryImpl>(),
                std::make_unique<IUs4OEMFactoryImpl>(),
                std::make_unique<IUs4OEMInitializerImpl>(),
                std::make_unique<Us4RSettingsConverterImpl>()
            )
    );
}

int testConnection() {
    return 765;
}

SessionImpl::SessionImpl(const SessionSettings &sessionSettings,
                         Us4RFactory::Handle us4RFactory)
        : us4rFactory(std::move(us4RFactory)) {
    devices = configureDevices(sessionSettings);
}

Device::RawHandle SessionImpl::getDevice(const std::string &path) {
    // sanitize
    std::string sanitizedPath{path};
    boost::algorithm::trim(sanitizedPath);

    // parse path
    auto [root, tail] = ::arrus::getPathRoot(sanitizedPath);

    auto deviceId = DeviceId::parse(root);
    Device::RawHandle rootDevice = getDevice(deviceId);

    if(tail.empty()) {
        return rootDevice;
    } else {
        if(isInstanceOf<DeviceWithComponents>(rootDevice)) {
            return ((DeviceWithComponents*)rootDevice)->getDevice(tail);
        } else {
            throw IllegalArgumentException(arrus::format(
                "Invalid path '{}', top-level devices can be accessed only.",
                path
            ));
        }
    }
}

Device::RawHandle SessionImpl::getDevice(const DeviceId &deviceId) {
    try {
        return devices.at(deviceId).get();
    } catch(const std::out_of_range&) {
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

}