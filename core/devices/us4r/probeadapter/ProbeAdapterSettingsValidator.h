#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERSETTINGSVALIDATOR_H

#include <unordered_set>
#include <range/v3/view/transform.hpp>
#include <limits>
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"

#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/DeviceSettingsValidator.h"
#include "arrus/core/common/collections.h"
#include "arrus/common/format.h"
#include "arrus/common/asserts.h"

namespace arrus {

class ProbeAdapterSettingsValidator
        : public DeviceSettingsValidator<ProbeAdapterSettings> {

public:
    explicit ProbeAdapterSettingsValidator(const Ordinal ordinal)
            : DeviceSettingsValidator(
            DeviceId(DeviceType::ProbeAdapter, ordinal)) {}

    void validate(const ProbeAdapterSettings &obj) override {
        using OEMMapping = Us4OEMSettings::ChannelMapping;
        using OEMMappingElement = OEMMapping::value_type;
        const auto N_RX = Us4OEMImpl::N_RX_CHANNELS;
        // Make sure, that the number of channels is equal to
        // the number of channel mapping elements.
        expectEqual<ChannelIdx>("channel mapping",
								static_cast<ChannelIdx>(obj.getChannelMapping().size()),
                                obj.getNumberOfChannels(),
                                " (size, compared to nChannels)");

        // Get the number of us4oems
        auto modules = obj.getChannelMapping() |
                       ranges::view::transform([](auto v) { return v.first; });

        // Check if contains consecutive ordinal numbers.
        // Us4OEMs should have exactly ordinals: 0, 1, ... nmodules-1
        std::unordered_set<Ordinal> modulesSet{std::begin(modules),
                                               std::end(modules)};
        ARRUS_REQUIRES_TRUE(
                modulesSet.size() >= std::numeric_limits<Ordinal>::min()
                && modulesSet.size() <= std::numeric_limits<Ordinal>::max(),
                arrus::format("Us4OEMs count should be in range {}, {}",
                              std::numeric_limits<Ordinal>::min(),
                              std::numeric_limits<Ordinal>::max()));

        for(Ordinal i = 0; i < (Ordinal) modulesSet.size(); ++i) {
            expectTrue("channel mapping",
                       setContains(modulesSet, i),
                       arrus::format("Missing Us4OEM: {}", i));
        }

        if(hasErrors()) {
            // Do not continue here, some errors may impact further validation
            // correctness.
            return;
        }

        auto nModules = modulesSet.size();

        // Split to us4oem channel mappings
        std::vector<OEMMapping> us4oemMappings(nModules);

        for(auto[module, channel] : obj.getChannelMapping()) {
            us4oemMappings[module].emplace_back(channel);
        }

        unsigned us4oemOrdinal = 0;

        for(auto &mapping : us4oemMappings) {
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
                auto groups = channelsSet | ranges::view::transform(
                        [=](auto v) { return v / N_RX; });


                bool isSingleGroup = std::reduce(
                        std::begin(groups), std::end(groups),
                        true,
                        [&groups](bool init, auto idx) {
                           return init && idx == groups.front();
                        });

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
