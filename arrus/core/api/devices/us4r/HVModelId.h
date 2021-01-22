#ifndef ARRUS_CORE_API_DEVICES_US4R_HVMODELID_H
#define ARRUS_CORE_API_DEVICES_US4R_HVMODELID_H

#include <string>
#include <sstream>
#include <utility>

namespace arrus::devices {

class HVModelId {

public:
    HVModelId(std::string manufacturer, std::string name)
        : manufacturer(std::move(manufacturer)),
          name(std::move(name)) {}

    const std::string &getManufacturer() const {
        return manufacturer;
    }

    const std::string &getName() const {
        return name;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const HVModelId &id) {
        os << "HVModelId("
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

#endif //ARRUS_CORE_API_DEVICES_US4R_HVMODELID_H
