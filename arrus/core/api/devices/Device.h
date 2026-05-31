#ifndef ARRUS_CORE_API_DEVICES_DEVICE_H
#define ARRUS_CORE_API_DEVICES_DEVICE_H

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/Parameters.h"
#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

#include <memory>
#include <string>

namespace arrus::devices {
class Device {
public:
    using Handle = std::unique_ptr<Device>;
    using RawHandle = PtrHandle<Device>;

    DeviceId getDeviceId() const {
        return id;
    }

    /**
     * Returns description of the device. Virtual ABI surface uses
     * std4us::String for stability across MSVC Debug/Release.
     */
    virtual std4us::String getDescriptionNative() const {
        return std4us::String("Unknown Device");
    }

    /**
     * Backward-compatibility shim (non-virtual) returning std::string by
     * value. The std::string is constructed in the caller's TU.
     */
    std::string getDescription() const {
        return std4us::toStd(getDescriptionNative());
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
