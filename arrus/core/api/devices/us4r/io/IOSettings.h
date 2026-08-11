#ifndef ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H

#include <utility>
#include <unordered_map>
#include <algorithm>

#include "IOCapability.h"
#include "IOAddress.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus::devices::us4r {

/**
 * Us4R IO settings (capabilities, etc.).
 *
 * This class specifies the mapping from IO address (IO ordinal number) to IO capability.
 *
 * Please use \ref IOSettingsBuilder to create an instance of this class.
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

    bool hasFrameMetadataCapability() const {
        auto it = addresses.find(IOCapability::FRAME_METADATA);
        return it != std::end(addresses) && it->second.size() > 0;
    }

    IOAddressSet getFrameMetadataCapabilityAddresses() const {
        if(!hasFrameMetadataCapability()) {
            throw arrus::IllegalArgumentException("The IO Settings of the device have no pulse counter capability.");
        }
        return addresses.find(IOCapability::FRAME_METADATA)->second;
    }

    std::unordered_set<Ordinal> getFrameMetadataCapabilityOEMs() const {
        auto addrs = getFrameMetadataCapabilityAddresses();
        // Check if only a single OEM is addressed by pulse counter functionality.
        std::unordered_set<Ordinal> oems;
        for(auto &a: addrs) {
            oems.insert(a.getUs4OEM());
        }
        return oems;
    }

    bool hasSequenceTriggerCapability() const {
        auto it = addresses.find(IOCapability::SEQUENCE_TRIGGER);
        return it != std::end(addresses) && it->second.size() > 0;
    }

    IOAddress getSequenceTriggerCapabilityAddress() const {
        if(!hasSequenceTriggerCapability()) {
            throw arrus::IllegalArgumentException("The IO Settings of the device have no sequence trigger capability.");
        }
        const auto &addrs = addresses.find(IOCapability::SEQUENCE_TRIGGER)->second;
        if(addrs.size() != 1) {
            throw arrus::IllegalArgumentException("The sequence trigger capability accepts only a single address.");
        }
        return *addrs.begin();
    }

private:
    std::unordered_map<IOCapability, IOAddressSet> addresses;
};

/**
 * IO Settings builder.
 */
class IOSettingsBuilder {
public:
    /**
     * Assigns probe-connected check capability to the given (us4OEM, IO number).
     */
    IOSettingsBuilder &setProbeConnectedCheckCapability(const IOAddressSet& addresses) {
        addr.emplace(IOCapability::PROBE_CONNECTED_CHECK, addresses);
        return *this;
    }

    /**
     * Assigns frame-metadata capability to the given (us4OEM, IO number).
     *
     * NOTE: frame-metadata capability (e.g. external encoder metadata) requires us4OEM custom firmware.
     */
    IOSettingsBuilder &setFrameMetadataCapability(const IOAddressSet& addresses) {
        addr.emplace(IOCapability::FRAME_METADATA, addresses);
        return *this;
    }

    IOSettingsBuilder &setSequenceTriggerCapability(const IOAddressSet& addresses) {
        if(addresses.size() != 1) {
            throw arrus::IllegalArgumentException("The sequence trigger capability accepts only a single address.");
        }
        addr.emplace(IOCapability::SEQUENCE_TRIGGER, addresses);
        return *this;
    }

    /**
     * Builds the IOSettings for the selected IO - capability mappings.
     */
    IOSettings build() {
        return IOSettings(addr);
    }

private:
    std::unordered_map<IOCapability, IOAddressSet> addr;
};

}


#endif//ARRUS_CORE_API_DEVICES_US4R_IO_IOSETTINGS_H
