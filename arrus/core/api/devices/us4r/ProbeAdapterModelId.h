#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H

#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

namespace arrus::devices {

class ProbeAdapterModelId {
public:
    explicit ProbeAdapterModelId(const std::string &manufacturer, const std::string &name)
        : manufacturer(std4us::fromStd(manufacturer)), name(std4us::fromStd(name)) {}

    explicit ProbeAdapterModelId(std4us::String manufacturer, std4us::String name)
        : manufacturer(std::move(manufacturer)), name(std::move(name)) {}

    std::string getName() const { return std4us::toStd(name); }
    std::string getManufacturer() const { return std4us::toStd(manufacturer); }

    const std4us::String &getNameNative() const { return name; }
    const std4us::String &getManufacturerNative() const { return manufacturer; }

    friend std::ostream &operator<<(std::ostream &os, const ProbeAdapterModelId &id) {
        os << "ProbeAdapterModelId("
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

#endif//ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H
