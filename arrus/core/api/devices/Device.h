#ifndef ARRUS_CORE_API_DEVICES_DEVICE_H
#define ARRUS_CORE_API_DEVICES_DEVICE_H

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/Parameters.h"

#include <memory>

namespace arrus::devices {
class Device {
public:
    using Handle = std::unique_ptr<Device>;
    using RawHandle = PtrHandle<Device>;

    DeviceId getDeviceId() const {
        return id;
    }

    virtual ~Device() = default;

    virtual void setParameters(const Parameters &) {
        throw IllegalArgumentException("This device does not support setting any parameters.");
    }

protected:
    explicit Device(const DeviceId &id): id(id) {}

    DeviceId id;
};

}




#endif //ARRUS_CORE_API_DEVICES_DEVICE_H
