#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H

#include <string>
#include <utility>

namespace arrus {

class ProbeModelId {
public:
    explicit ProbeModelId(std::string name) : name(std::move(name)) {}

    [[nodiscard]] const std::string &getName() const {
        return name;
    }

    [[nodiscard]] const std::string &getManufacturer() const {
        return manufacturer;
    }

private:
    std::string name;
    std::string manufacturer;
};

}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBEMODELID_H
