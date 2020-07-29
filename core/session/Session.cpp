#include "Session.h"

#include "core/common/format.h"
#include "boost/algorithm/string.hpp"

namespace arrus {

Session::Session(const SessionSettings &sessionSettings) {
    devices = configureDevices(sessionSettings.getSystemSettings());
}

DeviceHandle &Session::getDevice(const std::string &deviceId) {
}

DeviceHandle &Session::getDevice(const DeviceId &deviceId) {
    try {
        return devices.at(deviceId);
    } catch (const std::out_of_range &e) {
        throw IllegalArgumentException(
                arrus::format("Unrecognized device: {}", deviceId.toString()));
    }
}

Session::DeviceMap Session::configureDevices(const SystemSettings &settings) {
    DeviceMap result;
}

}