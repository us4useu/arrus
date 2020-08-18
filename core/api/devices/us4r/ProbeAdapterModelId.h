#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H

#include <string>
#include <utility>

namespace arrus {

class ProbeAdapterModelId {
public:
    explicit ProbeAdapterModelId(std::string name, std::string manufacturer)
            : name(std::move(name)), manufacturer(manufacturer) {}

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

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERMODELID_H
