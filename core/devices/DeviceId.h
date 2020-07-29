#ifndef ARRUS_CORE_DEVICES_DEVICEID_H
#define ARRUS_CORE_DEVICES_DEVICEID_H

#include <sstream>
#include <unordered_map>

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
} DeviceType;

static const std::unordered_map<DeviceTypeEnum, std::string>
        DEVICE_TYPE_ENUM_STRINGS = {
        {Us4OEM,              "Us4OEM"},
        {UltrasoundInterface, "UltrasoundInterface"},
        {Probe,               "Probe"},
        {GPU,                 "GPU"},
        {CPU,                 "CPU"},
};

/**
 * Converts string to DeviceTypeEnum.
 *
 * @param deviceTypeStr string representation of device type enum.
 * @return device type enum
 */
DeviceTypeEnum parseToDeviceTypeEnum(const std::string &deviceTypeStr);

/**
 * Converts DeviceTypeEnum to string.
 *
 * @param deviceTypeEnum device type enum to convert
 * @return string representation of device type
 */
std::string toString(DeviceTypeEnum deviceTypeEnum);

/**
 * Device ordinal number, e.g. GPU 0, GPU 1, Us4OEM 0, Us4OEM 1 etc.
 */
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

    friend std::ostream &operator<<(std::ostream &os, const DeviceId &id);

    [[nodiscard]] std::string toString() const {
        std::ostringstream ss;
        ss << *this;
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
