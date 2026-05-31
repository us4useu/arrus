#ifndef ARRUS_CORE_API_DEVICES_DEVICEID_H
#define ARRUS_CORE_API_DEVICES_DEVICEID_H

#include <sstream>
#include <string>

#include "arrus/core/api/common/macros.h"
#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

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
 * Converts string to DeviceType. Native (exported) overload takes
 * std4us::String for ABI stability.
 */
ARRUS_CPP_EXPORT
DeviceType parseToDeviceTypeEnumNative(const std4us::String &deviceTypeStr);

/** Backward-compatibility shim accepting std::string (inline). */
inline DeviceType parseToDeviceTypeEnum(const std::string &deviceTypeStr) {
    return parseToDeviceTypeEnumNative(std4us::fromStd(deviceTypeStr));
}

/**
 * Converts DeviceType to string. Native (exported) overload returns
 * std4us::String for ABI stability.
 */
ARRUS_CPP_EXPORT
std4us::String toStringNative(DeviceType deviceTypeEnum);

/** Backward-compatibility shim returning std::string (inline). */
inline std::string toString(DeviceType deviceTypeEnum) {
    return std4us::toStd(toStringNative(deviceTypeEnum));
}

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
    static DeviceId parseNative(const std4us::String &deviceId);

    /** Backward-compatibility shim (inline). */
    static DeviceId parse(const std::string &deviceId) {
        return parseNative(std4us::fromStd(deviceId));
    }

private:
    DeviceType deviceType;
    Ordinal ordinal;
};

}

#endif //ARRUS_CORE_API_DEVICES_DEVICEID_H
