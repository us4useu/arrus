#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSVALIDATOR_H

#include <unordered_set>
#include <limits>
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"

#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/SettingsValidator.h"
#include "arrus/core/common/collections.h"
#include "arrus/common/format.h"
#include "arrus/common/asserts.h"

namespace arrus::devices {

class ProbeAdapterSettingsValidator
        : public SettingsValidator<ProbeAdapterSettings> {

public:
    explicit ProbeAdapterSettingsValidator(const Ordinal ordinal)
            : SettingsValidator(DeviceId(DeviceType::ProbeAdapter, ordinal)) {}

    void validate(const ProbeAdapterSettings &obj) override {
        auto &id = obj.getModelId();
        expectTrue("modelId", !id.getManufacturer().empty(),
                   "manufacturer name should not be empty.");
        expectTrue("modelId", !id.getName().empty(),
                   "device name should not be empty.");

        using OEMMapping = Us4OEMSettings::ChannelMapping;
        using OEMMappingElement = OEMMapping::value_type;
        const auto N_RX = Us4OEMImpl::N_RX_CHANNELS;
        // Make sure, that the number of channels is equal to
        // the number of channel mapping elements.
        expectEqual<ChannelIdx>("channel mapping", static_cast<ChannelIdx>(obj.getChannelMapping().size()),
                                obj.getNumberOfChannels(),
                                " (size, compared to nChannels)");

        // Get the number of us4oems
        std::set<Ordinal> modules;
        for(auto &moduleChannel : obj.getChannelMapping()) {
            auto us4oem = moduleChannel.first;
            modules.insert(us4oem);
        }
        auto maxModuleNr = *std::max_element(std::begin(modules), std::end(modules));
        // Split to us4oem channel mappings
        std::vector<OEMMapping> us4oemMappings(maxModuleNr+1);

        for(auto[module, channel] : obj.getChannelMapping()) {
            us4oemMappings[module].emplace_back(channel);
        }

        unsigned us4oemOrdinal = 0;
        for(auto &mapping : us4oemMappings) {
            if(mapping.empty()) {
                // This module is not used and has no mapping available.
                continue;
            }
            // Make sure that the number of channels for each module is
            // multiple of nRx
            expectDivisible("channel mapping",
                            (ChannelIdx) mapping.size(), N_RX,
                            arrus::format(" size (for Us4OEM: {})", us4oemOrdinal));

            auto nIt = mapping.size() / N_RX;
            for(unsigned i = 0; i < nIt; ++i) {
                std::unordered_set<OEMMappingElement> channelsSet(
                        std::begin(mapping) + i * N_RX,
                        std::begin(mapping) + (i + 1) * N_RX);

                // Make sure that the channel mappings are unique in given groups.
                expectEqual(
                        "channel mapping",
                        (ChannelIdx) channelsSet.size(), N_RX,
                        arrus::format(
                                " (number of unique channel indices "
                                "for Us4OEM: {}, "
                                "for input range [{}, {}))",
                                us4oemOrdinal, i*N_RX, (i+1)*N_RX));

                // Make sure, the mapping contains [0,31)*i*32 groups
                // (where i can be 0, 1, 2, 3)
                std::unordered_set<OEMMappingElement> groups;
                for(auto v: channelsSet) {
                    groups.insert(v / N_RX);
                }
                bool isSingleGroup = groups.size() == 1;

                expectTrue(
                        "channel mapping",
                        isSingleGroup,
                        arrus::format(
                                "(for Us4OEM:{}): "
                                "Channels [{}, {}] should be in the single group "
                                "of 32 elements (i*32, (i+1)*32), where i "
                                "can be 0, 1, 2, 3,....",
                                us4oemOrdinal, i * N_RX, (i + 1) * N_RX));
            }
            ++us4oemOrdinal;
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSVALIDATOR_H
