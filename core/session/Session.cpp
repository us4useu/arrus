#include "Session.h"

#include <boost/algorithm/string.hpp>

#include "core/common/format.h"

// Construction components.
#include "core/devices/us4oem/impl/Us4OEMFactoryImpl.h"
#include "core/devices/us4oem/impl/ius4oem/IUs4OEMFactoryImpl.h"

namespace arrus {

Session::Session(const SessionSettings &sessionSettings) {
    devices = configureDevices(sessionSettings.getSystemSettings());
}

Device::Handle &Session::getDevice(const std::string &path) {
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
    return getDevice(deviceId);
}

Device::Handle &Session::getDevice(const DeviceId &deviceId) {
    try {
        return devices.at(deviceId);
    } catch (const std::out_of_range &e) {
        throw IllegalArgumentException(
                arrus::format("Unrecognized device: {}", deviceId.toString()));
    }
}

Session::DeviceMap Session::configureDevices(const SystemSettings &settings) {
    DeviceMap result;

    // Us4OEMs.
    Us4OEMFactoryImpl us4oemFactory(IUs4OEMFactoryImpl::getInstance());

    for (auto &[ordinal, cfg] : settings.getUs4oemSettings()) {
        Us4OEM::Handle handle = us4oemFactory.getUs4OEM(ordinal, cfg);
        result.emplace(handle->getDeviceId(), std::move(handle));
    }

    // Adapters

    // Probes

    return result;
}


}