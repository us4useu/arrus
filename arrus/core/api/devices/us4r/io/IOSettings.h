#ifndef ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H

#include <utility>
#include <unordered_map>

#include "IOCapability.h"
#include "IOAddress.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus::devices::us4r {

/**
 * Us4R IO settings (capabilities, etc.)
 */
class IOSettings {
public:
    IOSettings() = default;

    explicit IOSettings(std::unordered_map<IOCapability, IOAddressSet> addresses) : addresses(std::move(addresses)) {}

    bool hasProbeConnectedCheckCapability() const {
        auto it = addresses.find(IOCapability::PROBE_CONNECTED_CHECK);
        return it != std::end(addresses) && it->second.size() > 0;
    }

    IOAddress getProbeConnectedCheckCapabilityAddress() const {
        if(!hasProbeConnectedCheckCapability()) {
            throw arrus::IllegalArgumentException("The IO Settings of the device have no probe connected check "
                                                  "capability.");
        }
        return *addresses.find(IOCapability::PROBE_CONNECTED_CHECK)->second.begin();
    }

private:
    std::unordered_map<IOCapability, IOAddressSet> addresses;
};

/**
 * IO Settings builder.
 */
class IOSettingsBuilder {
public:
    IOSettingsBuilder &setProbeConnectedCheckCapability(const IOAddressSet& addresses) {
        addr.emplace(IOCapability::PROBE_CONNECTED_CHECK, addresses);
        return *this;
    }

    IOSettings build() {
        return IOSettings(addr);
    }

private:
    std::unordered_map<IOCapability, IOAddressSet> addr;
};

}


#endif//ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H
