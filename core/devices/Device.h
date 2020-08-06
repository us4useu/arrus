#ifndef ARRUS_CORE_FRAMEWORK_DEVICE_H
#define ARRUS_CORE_FRAMEWORK_DEVICE_H

#include "arrus/core/devices/DeviceId.h"

#include <memory>

namespace arrus {

class Device {
public:
    using Handle = std::unique_ptr<Device>;

    DeviceId getDeviceId() const {
        return id;
    }

protected:
    Device(const DeviceId &id): id(id) {}

    DeviceId id;
};

}




#endif //ARRUS_CORE_FRAMEWORK_DEVICE_H
