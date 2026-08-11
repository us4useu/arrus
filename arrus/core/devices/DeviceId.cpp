#include <boost/bimap.hpp>
#include <boost/lexical_cast.hpp>
#include <format>
#include <regex>
#include <std4us/string.h>

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/asserts.h"

namespace arrus::devices {

static const std::unordered_map<DeviceType, std::string>
        DEVICE_TYPE_ENUM_STRINGS = {
        {DeviceType::Us4R,        "Us4R"},
        {DeviceType::Us4OEM,      "Us4OEM"},
        {DeviceType::ProbeAdapter,"ProbeAdapter"},
        {DeviceType::Probe,       "Probe"},
        {DeviceType::GPU,         "GPU"},
        {DeviceType::CPU,         "CPU"},
        {DeviceType::HV,          "HV"},
        {DeviceType::Ultrasound,  "Ultrasound"},
        {DeviceType::File,        "File"},
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
    catch (const std::out_of_range&) {
        std::vector<std::string> availableKeys =
                DeviceTypeEnumStringRepr::getInstance().keys();
        std::sort(availableKeys.begin(), availableKeys.end());
        const auto availableKeysMsg =
                std4us::join(availableKeys, ", ");
        throw IllegalArgumentException(
                std::format("Unrecognized device type: {}, "
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
    std4us::split(deviceIdComponents, deviceId, ":");

    if (deviceIdComponents.size() != 2) {
        throw IllegalArgumentException(std::format(
                "Device id should be in the format of: deviceType:ordinal "
                "(got: '{}')", deviceId
        ));
    }
    auto deviceTypeStr = std4us::trim(deviceIdComponents[0]);
    auto ordinalStr = std4us::trim(deviceIdComponents[1]);
    // Device Type.
    DeviceType deviceTypeEnum = parseToDeviceTypeEnum(deviceTypeStr);

    // Device Ordinal.
    // Requires only digits in the ordinal part.
    ARRUS_REQUIRES_TRUE_FOR_ARGUMENT(std::regex_match(ordinalStr, std::regex("[0-9]+")),
            std::format("Invalid device number: {}", ordinalStr)
    );
    Ordinal ordinal;
    ARRUS_REQUIRES_NO_THROW(
            ordinal = boost::lexical_cast<Ordinal>(ordinalStr),
            boost::bad_lexical_cast,
            arrus::IllegalArgumentException(
                    std::format("Invalid device number: {}", ordinalStr)
                    )
    );
    return DeviceId(deviceTypeEnum, ordinal);
}

std::ostream& operator<<(std::ostream &os, const DeviceId &id) {
    os << toString(id.deviceType) << ":" << id.ordinal;
    return os;
}

}