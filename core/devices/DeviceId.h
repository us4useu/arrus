#ifndef ARRUS_CORE_DEVICES_DEVICEID_H
#define ARRUS_CORE_DEVICES_DEVICEID_H

#include <sstream>
#include "core/common/hash.h"

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
    // Remember to update DeviceTypeEnumStringRepr.
} DeviceType;

/**
 * Converts string to DeviceTypeEnum.
 *
 * @param deviceTypeStr string representation of device type enum.
 * @return device type enum
 */
DeviceTypeEnum parseToDeviceTypeEnum(const std::string& deviceTypeStr);

/**
 * Converts DeviceTypeEnum to string.
 *
 * @param deviceTypeEnum device type enum to convert
 * @return string representation of device type
 */
std::string toString(DeviceTypeEnum deviceTypeEnum);


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

    friend std::ostream &operator<<(std::ostream &os, const DeviceId &id) {
        os << "Device " << id.deviceType << ":" << id.ordinal;
        return os;
    }

    friend std::string to_string(const DeviceId& id) {
        std::ostringstream ss;
        ss << id;
        return ss.str();
    }

    static DeviceId parse(const std::string &deviceId);

private:
    DeviceType deviceType;
    Ordinal ordinal;
};

MAKE_HASHER(DeviceId, t.getDeviceType(), t.getOrdinal())
}

#endif //ARRUS_CORE_DEVICES_DEVICEID_H
