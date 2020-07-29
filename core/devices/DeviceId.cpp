#include "DeviceId.h"

#include <boost/bimap.hpp>
#include <boost/algorithm/string.hpp>

#include "core/common/exceptions.h"
#include "core/common/format.h"

namespace arrus {

/**
 * String representation of Device Type.
 */
class DeviceTypeEnumStringRepr {
public:
    DeviceTypeEnumStringRepr(const DeviceTypeEnumStringRepr &) = delete;

    void operator=(const DeviceTypeEnumStringRepr &) = delete;

    static DeviceTypeEnumStringRepr &getInstance() {
        static DeviceTypeEnumStringRepr instance;
        return instance;
    }

    std::string toString(const DeviceTypeEnum deviceTypeEnum) {
        return reprs.right.at(deviceTypeEnum);
    }

    DeviceTypeEnum parse(const std::string &deviceTypeStr) {
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
        typedef boost::bimap<std::string, DeviceTypeEnum>::value_type val;
        reprs.insert(val("Us4OEM", DeviceTypeEnum::Us4OEM));
        reprs.insert(val("UltrasoundInterface",
                         DeviceTypeEnum::UltrasoundInterface));
        reprs.insert(val("Probe", DeviceTypeEnum::Probe));
        reprs.insert(val("GPU", DeviceTypeEnum::GPU));
        reprs.insert(val("CPU", DeviceTypeEnum::CPU));
    }

    boost::bimap<std::string, DeviceTypeEnum> reprs;
};

DeviceTypeEnum parseToDeviceTypeEnum(const std::string &deviceTypeStr) {
    try {
        return DeviceTypeEnumStringRepr::getInstance().parse(deviceTypeStr);
    }
    catch (const std::out_of_range &e) {
        std::vector<std::string> availableKeys = DeviceTypeEnumStringRepr::getInstance().keys();
        std::sort(availableKeys.begin(), availableKeys.end());
        const auto availableKeysMsg = boost::algorithm::join(availableKeys,
                                                             ", ");
        throw IllegalArgumentException(
                arrus::format("Unrecognized device type: {}, "
                              "allowed types: {}", deviceTypeStr,
                              availableKeysMsg));
    }
}

std::string toString(const DeviceTypeEnum deviceTypeEnum) {
    return DeviceTypeEnumStringRepr::getInstance().toString(deviceTypeEnum);
}

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
    DeviceTypeEnum deviceTypeEnum = parseToDeviceTypeEnum(deviceTypeStr);
    unsigned int ordinal = arrus::stoui(ordinalStr);
    return DeviceId(deviceTypeEnum, ordinal);
}

}