#ifndef ARRUS_CORE_DEVICES_TYPES_H
#define ARRUS_CORE_DEVICES_TYPES_H

namespace arrus {

/**
 * Device types available in the system.
 */
enum DeviceType {
    Us4OEM,
    UltrasoundInterface,
    Probe,
    GPU,
    CPU
};

using Ordinal = unsigned short;

/**
 * Device identifier.
 */
class DeviceId {
    DeviceId(const DeviceType dt,
             const Ordinal ordinal = std::nullopt_t)
            : deviceType(dt), ordinal(ordinal)

    DeviceType getDeviceType() {
        return deviceType;
    }

    Ordinal getOrdinal() {
        return ordinal;
    }

private:
    DeviceType deviceType;
    Ordinal ordinal;
};

using TGCCurve = std::vector<float>;

}

#endif //ARRUS_CORE_DEVICES_TYPES_H
