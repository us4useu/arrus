#ifndef ARRUS_CORE_API_DEVICES_DEVICEWITHCOMPONENTS_H
#define ARRUS_CORE_API_DEVICES_DEVICEWITHCOMPONENTS_H

#include <string>

#include "arrus/core/api/devices/Device.h"

namespace arrus::devices {

class DeviceWithComponents {
public:
    /**
     * Returns a raw handle to the component of this device.
     *
     * @param path path to the component
     * @return a handle to the component
     */
    virtual Device::RawHandle getDevice(const std::string& path) = 0;
};

}

#endif //ARRUS_CORE_API_DEVICES_DEVICEWITHCOMPONENTS_H
