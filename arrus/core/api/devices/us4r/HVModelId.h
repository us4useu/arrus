#ifndef ARRUS_CORE_API_DEVICES_US4R_HVMODELID_H
#define ARRUS_CORE_API_DEVICES_US4R_HVMODELID_H

#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

namespace arrus::devices {

class HVModelId {

public:
    HVModelId(const std::string &manufacturer, const std::string &name)
        : manufacturer(std4us::fromStd(manufacturer)), name(std4us::fromStd(name)) {}

    HVModelId(std4us::String manufacturer, std4us::String name)
        : manufacturer(std::move(manufacturer)), name(std::move(name)) {}

    std::string getManufacturer() const { return std4us::toStd(manufacturer); }
    std::string getName() const { return std4us::toStd(name); }

    const std4us::String &getManufacturerNative() const { return manufacturer; }
    const std4us::String &getNameNative() const { return name; }

    friend std::ostream &operator<<(std::ostream &os, const HVModelId &id) {
        os << "HVModelId("
           << "manufacturer: " << id.manufacturer.c_str()
           << " name: " << id.name.c_str()
           << ")";
        return os;
    }

    std::string toString() const {
        std::stringstream sstr;
        sstr << *this;
        return sstr.str();
    }

private:
    std4us::String manufacturer;
    std4us::String name;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_US4R_HVMODELID_H
