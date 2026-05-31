#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H

#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

namespace arrus::devices {

/**
 * Storage uses std4us::String for ABI stability across MSVC Debug/Release.
 * std::string-typed constructors and accessors remain as inline header-only
 * backward-compatibility shims; std::string instances never cross the
 * library boundary.
 */
class ProbeModelId {
public:
    explicit ProbeModelId(const std::string &manufacturer, const std::string &name)
        : manufacturer(std4us::fromStd(manufacturer)), name(std4us::fromStd(name)) {}

    explicit ProbeModelId(std4us::String manufacturer, std4us::String name)
        : manufacturer(std::move(manufacturer)), name(std::move(name)) {}

    std::string getName() const { return std4us::toStd(name); }

    std::string getManufacturer() const { return std4us::toStd(manufacturer); }

    const std4us::String &getNameNative() const { return name; }

    const std4us::String &getManufacturerNative() const { return manufacturer; }

    friend std::ostream &operator<<(std::ostream &os, const ProbeModelId &id) {
        os << "ProbeModel("
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

#endif//ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
