#ifndef ARRUS_CORE_API_DEVICES_DEVICEWITHCOMPONENTS_H
#define ARRUS_CORE_API_DEVICES_DEVICEWITHCOMPONENTS_H

#include <string>

#include "arrus/core/api/devices/Device.h"
#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

namespace arrus::devices {

class DeviceWithComponents {
public:
    /**
     * Returns a raw handle to the component of this device. Virtual ABI
     * surface uses std4us::String for stability across MSVC Debug/Release.
     *
     * @param path path to the component
     * @return a handle to the component
     */
    virtual Device::RawHandle getDeviceNative(const std4us::String &path) = 0;

    /** Backward-compatibility shim (non-virtual). */
    Device::RawHandle getDevice(const std::string &path) {
        return getDeviceNative(std4us::fromStd(path));
    }
};

}// namespace arrus::devices

#endif //ARRUS_CORE_API_DEVICES_DEVICEWITHCOMPONENTS_H
