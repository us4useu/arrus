#ifndef ARRUS_CORE_API_DEVICES_DEVICEID_H
#define ARRUS_CORE_API_DEVICES_DEVICEID_H

#include <sstream>
#include <string>

#include "arrus/core/api/common/macros.h"

namespace arrus::devices {

/**
 * Device types available in the system.
 */
enum class DeviceType {
    Us4R,
    Us4OEM,
    ProbeAdapter,
    Probe,
    GPU,
    CPU,
    HV,
    Ultrasound,
    File,
};

/**
    * Converts string to DeviceType.
    *
    * @param deviceTypeStr string representation of device type enum.
    * @return device type enum
    */
ARRUS_CPP_EXPORT
DeviceType parseToDeviceTypeEnum(const std::string &deviceTypeStr);

/**
 * Converts DeviceType to string.
 *
 * @param deviceType device type enum to convert
 * @return string representation of device type
 */
ARRUS_CPP_EXPORT
std::string toString(DeviceType deviceTypeEnum);

/**
 * Device ordinal number, e.g. GPU 0, GPU 1, Us4OEM 0, Us4OEM 1 etc.
 */
using Ordinal = unsigned short;

/**
 * Device identifier.
 */
class DeviceId {
public:
    DeviceId(const DeviceType dt, const Ordinal ordinal) : deviceType(dt), ordinal(ordinal) {}

    DeviceType getDeviceType() const {
        return deviceType;
    }

    Ordinal getOrdinal() const {
        return ordinal;
    }

    bool operator==(const DeviceId &rhs) const {
        return deviceType == rhs.deviceType
               && ordinal == rhs.ordinal;
    }

    bool operator!=(const DeviceId &rhs) const {
        return !(rhs == *this);
    }

    ARRUS_CPP_EXPORT
    friend std::ostream &operator<<(std::ostream &os, const DeviceId &id);

    std::string toString() const {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    ARRUS_CPP_EXPORT
    static DeviceId parse(const std::string &deviceId);

private:
    DeviceType deviceType;
    Ordinal ordinal;
};

}

#endif //ARRUS_CORE_API_DEVICES_DEVICEID_H
