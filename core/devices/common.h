#ifndef ARRUS_CORE_DEVICES_TYPES_H
#define ARRUS_CORE_DEVICES_TYPES_H

#include <vector>

#include "Device.h"

namespace arrus {

/**
 * Device types available in the system.
 */
typedef enum DeviceTypeEnum {
    Us4OEM,
    UltrasoundInterface,
    Probe,
    GPU,
    CPU
} DeviceType;

using Ordinal = unsigned short;

/**
 * Device identifier.
 */
class DeviceId {
    DeviceId(const DeviceType dt,
             const Ordinal ordinal)
            : deviceType(dt), ordinal(ordinal)
    {}

    DeviceType getDeviceType() const {
        return deviceType;
    }

    Ordinal getOrdinal() const {
        return ordinal;
    }

private:
    DeviceType deviceType;
    Ordinal ordinal;
};

using DeviceHandle = std::shared_ptr<Device>;
using TGCCurve = std::vector<float>;

}

#endif //ARRUS_CORE_DEVICES_TYPES_H
