#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H

#include <string>
#include <utility>
#include <ostream>
#include <sstream>

namespace arrus {

class ProbeModelId {
public:
    explicit ProbeModelId(std::string name, std::string manufacturer)
    : name(std::move(name)), manufacturer(std::move(manufacturer)) {}

    [[nodiscard]] const std::string &getName() const {
        return name;
    }

    [[nodiscard]] const std::string &getManufacturer() const {
        return manufacturer;
    }

    friend std::ostream &operator<<(std::ostream &os, const ProbeModelId &id) {
        os << "ProbeModel("
           << "name: " << id.name << " manufacturer: " << id.manufacturer
           << ")";
        return os;
    }

    [[nodiscard]] std::string toString() const {
        std::stringstream sstr;
        sstr << *this;
        return sstr.str();
    }

private:
    std::string name;
    std::string manufacturer;
};

}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
