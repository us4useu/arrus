#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H

#include <string>
#include <utility>
#include <ostream>
#include <sstream>

namespace arrus::devices {

class ProbeModelId {
public:
    explicit ProbeModelId(std::string manufacturer, std::string name)
    : manufacturer(std::move(manufacturer)), name(std::move(name)) {}

    const std::string &getName() const {
        return name;
    }

    const std::string &getManufacturer() const {
        return manufacturer;
    }

    friend std::ostream &operator<<(std::ostream &os, const ProbeModelId &id) {
        os << "ProbeModel("
           << "manufacturer: " << id.manufacturer
           << " name: " << id.name
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

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
