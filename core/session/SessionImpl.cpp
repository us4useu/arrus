#include "arrus/core/session/SessionImpl.h"

#include <boost/algorithm/string.hpp>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/common/format.h"

// Construction components.
#include "arrus/core/devices/us4r/Us4RFactoryImpl.h"

namespace arrus {

SessionImpl::SessionImpl(const SessionSettings &sessionSettings) {
    devices = configureDevices(sessionSettings);
}

Device::Handle &SessionImpl::getDevice(const std::string &path) {
    // parse path
    // accept only the top-level devices
    std::vector<std::string> pathComponents;
    boost::algorithm::split(pathComponents, path,
                            boost::is_any_of("/"));

    if (pathComponents.size() != 1) {
        throw IllegalArgumentException(arrus::format(
                "Invalid path '{}', top-level devices can be accessed only.",
                path
        ));
    }
    auto deviceId = DeviceId::parse(pathComponents[0]);
    // If the top-level device is CompositeDevice, cast to it and call its getDevice
    // Consider handling it in more general way

    return getDevice(deviceId);
}

Device::Handle &SessionImpl::getDevice(const DeviceId &deviceId) {
    try {
        return devices.at(deviceId);
    } catch (const std::out_of_range &e) {
        throw IllegalArgumentException(
                arrus::format("Unrecognized device: {}", deviceId.toString()));
    }
}

SessionImpl::DeviceMap
SessionImpl::configureDevices(const SessionSettings &sessionSettings) {
    DeviceMap result;

    // Configuring Us4R.
    const Us4RSettings& us4RSettings = sessionSettings.getUs4RSettings();


    // Us4RFactory - initialize

    // Get all component devices

//    Us4OEMFactoryImpl us4oemFactory(IUs4OEMFactoryImpl::getInstance());
//
//    for (auto &[ordinal, cfg] : settings.getUs4oemSettings()) {
//        Us4OEM::Handle handle = us4oemFactory.getUs4OEM(ordinal, cfg);
//        result.emplace(handle->getDeviceId(), std::move(handle));
//    }

    // Adapters

    // Probes

    return result;
}

}