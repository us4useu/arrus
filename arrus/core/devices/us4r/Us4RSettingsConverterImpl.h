#ifndef ARRUS_CORE_DEVICES_US4R_US4SETTINGSCONVERTERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4SETTINGSCONVERTERIMPL_H

#include "arrus/common/asserts.h"
#include "arrus/common/utils.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/Us4RSettingsConverter.h"

namespace arrus::devices {

class Us4RSettingsConverterImpl : public Us4RSettingsConverter {
public:
    std::pair<std::vector<Us4OEMSettings>, ProbeAdapterSettings>
    convertToUs4OEMSettings(const ProbeAdapterSettings &probeAdapterSettings,
                            const ProbeSettings &probeSettings,
                            const RxSettings &rxSettings,
                            const std::vector<ChannelIdx> &channelsMask,
                            Us4OEMSettings::ReprogrammingMode reprogrammingMode) override {

        // Assumption:
        // for each module there is N_RX_CHANNELS*k elements in mapping
        // each group of N_RX_CHANNELS contains elements grouped to a single bucket (i*32, (i+1)*32)
        const auto &adapterSettingsMapping =
            probeAdapterSettings.getChannelMapping();
        const auto &probeSettingsMapping = probeSettings.getChannelMapping();

        // get number of us4oems from the probe adapter mapping
        // Determined based on ADAPTER MAPPINGS
        Ordinal nUs4OEMs = getNumberOfModules(adapterSettingsMapping);
        // Convert to list of us4oem mappings and active channel groups

        auto const nRx = Us4OEMImpl::N_RX_CHANNELS;
        auto const nTx = Us4OEMImpl::N_TX_CHANNELS;
        auto const actChSize = Us4OEMImpl::ACTIVE_CHANNEL_GROUP_SIZE;
        auto const nActChGroups = Us4OEMImpl::N_ACTIVE_CHANNEL_GROUPS;

        std::vector<Us4OEMSettings> result;
        std::vector<ChannelIdx> currentRxGroup(nUs4OEMs);
        std::vector<ChannelIdx> currentRxGroupElement(nUs4OEMs);

        // physical mapping for us4oems
        std::vector<Us4OEMSettings::ChannelMapping> us4oemChannelMapping;
        // logical mapping for adapter probe
        ProbeAdapterSettings::ChannelMapping adapterChannelMapping;

        // Initialize mappings with 0, 1, 2, 3, ... 127
        for(int i = 0; i < (int) nUs4OEMs; ++i) {
            Us4OEMSettings::ChannelMapping mapping(nTx);
            for(ChannelIdx j = 0; j < nTx; ++j) {
                mapping[j] = j;
            }
            us4oemChannelMapping.emplace_back(mapping);
        }
        // Map settings to:
        // - internal us4oem mapping,
        // - adapter channel mapping
        for(auto[module, channel] : probeAdapterSettings.getChannelMapping()) {
            // Channel mapping
            const auto group = channel / nRx;
            const auto element = currentRxGroupElement[module];
            if(element == 0) {
                // Starting new group
                currentRxGroup[module] = (ChannelIdx) group;
            } else {
                // Safety condition
                ARRUS_REQUIRES_TRUE(group == currentRxGroup[module],
                                    "Invalid probe adapter Rx channel mapping: "
                                    "inconsistent groups of channel "
                                    "(consecutive elements of N_RX_CHANNELS "
                                    "are required)");
            }
            auto logicalChannel = group * nRx + element;
            us4oemChannelMapping[module][logicalChannel] = channel;
            adapterChannelMapping.emplace_back(module,
                                               ChannelIdx(logicalChannel));

            currentRxGroupElement[module] =
                (currentRxGroupElement[module] + 1) % nRx;

        }

        // Active channel groups for us4oems
        std::vector<BitMask> activeChannelGroups;
        // Initialize masks
        for(Ordinal i = 0; i < nUs4OEMs; ++i) {
            // all groups turned off
            activeChannelGroups.emplace_back(nActChGroups);
        }
        for(const auto adapterChannel : probeSettingsMapping) {
            auto[module, logicalChannel] = adapterChannelMapping[adapterChannel];
            auto us4oemChannel = us4oemChannelMapping[module][logicalChannel];
            // When at least one channel in group has mapping, the whole
            // group of channels has to be active
            activeChannelGroups[module][us4oemChannel / actChSize] = true;
        }

        // CHANNELS MASKS PRODUCTION
        // convert probe channels masks to adapter channels mask:
        // - for each masked channel find the adapter's channel
        // convert adapter channels mask to us4oem channels mask
        // - for each masked channel n the adapter find module, channel
        // (note that we need use here logical channels of the us4oem
        // because we set tx apertures with logical channel (the physical channel is mapped by the IUs4OEM)

        std::vector<std::unordered_set<uint8>> channelsMasks(nUs4OEMs);

        // probe channel -> adapter channel -> [module, us4oem LOGICAL channel]
        for(const auto offProbeChannel : channelsMask) {
            ARRUS_REQUIRES_TRUE_E(offProbeChannel < probeSettings.getModel().getNumberOfElements().product(),
                                  ::arrus::IllegalArgumentException(
                                      ::arrus::format("Channels mask element {} cannot exceed "
                                                      "the number of probe elements {}",
                                                      offProbeChannel, probeSettings.getModel().
                                              getNumberOfElements().product())
                                  ));
            auto offAdapterChannel = probeSettingsMapping[offProbeChannel];
            auto[module, offUs4OEMLogicalChannel] = adapterChannelMapping[offAdapterChannel];
            channelsMasks[module].emplace(ARRUS_SAFE_CAST(offUs4OEMLogicalChannel, uint8));
        }
        // END OF THE MASKS PRODUCTION


        for(int i = 0; i < nUs4OEMs; ++i) {
            result.push_back(
                Us4OEMSettings(
                    us4oemChannelMapping[i],
                    activeChannelGroups[i],
                    rxSettings,
                    channelsMasks[i],
                    reprogrammingMode)
            );
        }
        return {result, ProbeAdapterSettings(
            probeAdapterSettings.getModelId(),
            probeAdapterSettings.getNumberOfChannels(),
            adapterChannelMapping
        )};
    }

private:
    static Ordinal
    getNumberOfModules(
        const ProbeAdapterSettings::ChannelMapping &adapterMapping) {
        std::vector<bool> mask(
            std::numeric_limits<Ordinal>::max());
        Ordinal count = 0;
        for(auto &moduleChannel : adapterMapping) {
            auto module = moduleChannel.first;
            if(!mask[module]) {
                count++;
                mask[module] = true;
            }
        }
        return count;
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4SETTINGSCONVERTERIMPL_H
