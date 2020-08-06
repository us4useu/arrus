#include <boost/bimap.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/common/format.h"
#include "arrus/core/common/asserts.h"

namespace arrus {

static const std::unordered_map<DeviceType, std::string>
        DEVICE_TYPE_ENUM_STRINGS = {
        {DeviceType::Us4R,        "Us4R"},
        {DeviceType::Us4OEM,      "Us4OEM"},
        {DeviceType::ProbeAdapter,"ProbeAdapter"},
        {DeviceType::Probe,       "Probe"},
        {DeviceType::GPU,         "GPU"},
        {DeviceType::CPU,         "CPU"}
};

/**
 * String representation of Device Type Enum.
 * Helper class to implement bi-directional translation enum -> string,
 * string -> enum.
 */
class DeviceTypeEnumStringRepr {
public:
    DeviceTypeEnumStringRepr(const DeviceTypeEnumStringRepr &) = delete;

    void operator=(const DeviceTypeEnumStringRepr &) = delete;

    static DeviceTypeEnumStringRepr &getInstance() {
        static DeviceTypeEnumStringRepr instance;
        return instance;
    }

    std::string toString(const DeviceType deviceTypeEnum) {
        return reprs.right.at(deviceTypeEnum);
    }

    DeviceType parse(const std::string &deviceTypeStr) {
        return reprs.left.at(deviceTypeStr);
    }

    std::vector<std::string> keys() {
        std::vector<std::string> result;

        std::transform(reprs.left.begin(), reprs.left.end(),
                       std::back_inserter(result),
                       [](auto const &p) { return p.first; });
        return result;
    }

private:
    DeviceTypeEnumStringRepr() {
        typedef boost::bimap<std::string, DeviceType>::value_type val;
        for (const auto& [e, str] : DEVICE_TYPE_ENUM_STRINGS) {
            reprs.insert({str, e});
        }
    }

    boost::bimap<std::string, DeviceType> reprs;
};

DeviceType parseToDeviceTypeEnum(const std::string &deviceTypeStr) {
    try {
        return DeviceTypeEnumStringRepr::getInstance().parse(deviceTypeStr);
    }
    catch (const std::out_of_range &e) {
        std::vector<std::string> availableKeys =
                DeviceTypeEnumStringRepr::getInstance().keys();
        std::sort(availableKeys.begin(), availableKeys.end());
        const auto availableKeysMsg =
                boost::algorithm::join(availableKeys,", ");
        throw IllegalArgumentException(
                arrus::format("Unrecognized device type: {}, "
                              "allowed types: {}", deviceTypeStr,
                              availableKeysMsg));
    }
}

std::string toString(const DeviceType deviceTypeEnum) {
    return DeviceTypeEnumStringRepr::getInstance().toString(deviceTypeEnum);
}

// DeviceId.

DeviceId DeviceId::parse(const std::string &deviceId) {
    std::vector<std::string> deviceIdComponents;
    boost::algorithm::split(deviceIdComponents, deviceId,
                            boost::is_any_of(":"));

    if (deviceIdComponents.size() != 2) {
        throw IllegalArgumentException(arrus::format(
                "Device id should be have format: deviceType:ordinal "
                "(got: '{}')", deviceId
        ));
    }
    auto deviceTypeStr = deviceIdComponents[0];
    auto ordinalStr = deviceIdComponents[1];
    boost::trim(deviceTypeStr);
    boost::trim(ordinalStr);
    // Device Type.
    DeviceType deviceTypeEnum = parseToDeviceTypeEnum(deviceTypeStr);

    // Device Ordinal.
    ARRUS_REQUIRES_TRUE_FOR_ARGUMENT(isDigitsOnly(ordinalStr),
            arrus::format("Invalid device number: {}", ordinalStr)
    );
    Ordinal ordinal;
    ARRUS_REQUIRES_NO_THROW(
            ordinal = boost::lexical_cast<Ordinal>(ordinalStr),
            boost::bad_lexical_cast,
            arrus::IllegalArgumentException(
                    arrus::format("Invalid device number: {}", ordinalStr)
                    )
    );
    return DeviceId(deviceTypeEnum, ordinal);
}

std::ostream& operator<<(std::ostream &os, const DeviceId &id) {
    os << toString(id.deviceType) << ":" << id.ordinal;
    return os;
}

}