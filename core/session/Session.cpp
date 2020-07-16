#include "Session.h"

namespace arrus {

Session::Session(const SessionSettings &sessionSettings) {
    devices = configureDevices(sessionSettings.getSystemSettings());
}

DeviceHandle Session::getDevice(const std::string &deviceId) {
    // TODO parse deviceId string
    // cal getDevice(DeviceId)
}

DeviceHandle Session::getDevice(const DeviceId &deviceId) {
    // TODO devices[deviceId]
}

Session::DeviceMap Session::configureDevices(const SystemSettings &settings) {
    DeviceMap deviceMap;
}

}