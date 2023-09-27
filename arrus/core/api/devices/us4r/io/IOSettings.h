#ifndef ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H

#include <utility>

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
        return addresses.find(IOCapability::PROBE_CONNECTED_CHECK) != std::end(addresses);
    }

    IOAddress getProbeConnectedCheckCapabilityAddress() const {
        return *addresses.find(IOCapability::PROBE_CONNECTED_CHECK)->second.begin();
    }

    bool hasPulseCounterCapability() const {
        return addresses.find(IOCapability::PULSE_COUNTER) != std::end(addresses);
    }

    IOAddressSet getPulseCounterCapabilityAddressSet() const {
        return addresses.find(IOCapability::PULSE_COUNTER)->second;
    }

private:
    std::unordered_map<IOCapability, IOAddressSet> addresses;
};

/**
 * IO Settings builder.
 */
class IOSettingsBuilder {

    IOSettingsBuilder &setProbeConnectedCheckCapability(const IOAddressSet& addresses) {
        addr.emplace(IOCapability::PROBE_CONNECTED_CHECK, addresses);
        return *this;
    }

    IOSettingsBuilder &setPulseCounterCapability(const IOAddressSet& addresses) {
        addr.emplace(IOCapability::PULSE_COUNTER, addresses);
        return *this;
    }

    IOSettings builder() {
        return IOSettings(addr);
    }

private:
    std::unordered_map<IOCapability, IOAddressSet> addr;
};

}


#endif//ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H
