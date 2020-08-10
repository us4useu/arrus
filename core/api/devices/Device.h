#ifndef ARRUS_CORE_API_DEVICE_H
#define ARRUS_CORE_API_DEVICE_H

#include "arrus/core/api/devices/DeviceId.h"

#include <memory>

namespace arrus {

class Device {
public:
    using Handle = std::unique_ptr<Device>;

    [[nodiscard]] DeviceId getDeviceId() const {
        return id;
    }

    virtual ~Device() = default;

protected:
    explicit Device(const DeviceId &id): id(id) {}

    DeviceId id;
};

}




#endif //ARRUS_CORE_API_DEVICE_H
