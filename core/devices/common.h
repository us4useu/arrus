#ifndef ARRUS_CORE_DEVICES_TYPES_H
#define ARRUS_CORE_DEVICES_TYPES_H

#include <vector>

#include "Device.h"
#include "core/utils/hash.h"

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
public:
    DeviceId(const DeviceType dt,
             const Ordinal ordinal)
            : deviceType(dt), ordinal(ordinal) {}

    [[nodiscard]] DeviceType getDeviceType() const {
        return deviceType;
    }

    [[nodiscard]] Ordinal getOrdinal() const {
        return ordinal;
    }

    bool operator==(const DeviceId &rhs) const {
        return deviceType == rhs.deviceType
               && ordinal == rhs.ordinal;
    }

    bool operator!=(const DeviceId &rhs) const {
        return !(rhs == *this);
    }

private:
    DeviceType deviceType;
    Ordinal ordinal;
};
MAKE_HASHER(DeviceId, t.getDeviceType(), t.getOrdinal())


using DeviceHandle = std::shared_ptr<Device>;
using TGCCurve = std::vector<float>;

}
#endif //ARRUS_CORE_DEVICES_TYPES_H
