#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H

#include <string>
#include <utility>
#include <ostream>

namespace arrus::devices {

class ProbeAdapterModelId {
public:
    explicit ProbeAdapterModelId(std::string manufacturer, std::string name)
        : manufacturer(std::move(manufacturer)), name(std::move(name)) {}

    const std::string &getName() const {
        return name;
    }

    const std::string &getManufacturer() const {
        return manufacturer;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const ProbeAdapterModelId &id) {
        os << "ProbeAdapterModelId("
        << "manufacturer: " << id.manufacturer << " name: " << id.name
        << ")";
        return os;
    }

    std::string toString() const {
        std::stringstream sstr;
        sstr << *this;
        return sstr.str();
    }

private:
    std::string manufacturer;
    std::string name;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H
