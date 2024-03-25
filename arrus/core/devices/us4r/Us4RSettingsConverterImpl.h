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
                            const RxSettings &rxSettings,
                            Us4OEMSettings::ReprogrammingMode reprogrammingMode,
                            std::optional<Ordinal> nUs4OEMsSetting,
                            const std::vector<Ordinal> &adapterToUs4RModuleNr,
                            int txFrequencyRange) override {
        typedef ProbeAdapterSettings PAS;
        // Assumption:
        // for each module there is N_RX_CHANNELS*k elements in mapping
        // each group of N_RX_CHANNELS contains elements grouped to a single bucket (i*32, (i+1)*32)
        PAS::ChannelMapping adapterMapping;
        Ordinal nUs4OEMs = 0;

        if(! adapterToUs4RModuleNr.empty()) {
            adapterMapping = remapUs4OEMs(probeAdapterSettings.getChannelMapping(), adapterToUs4RModuleNr);
        }
        else {
            adapterMapping = probeAdapterSettings.getChannelMapping();
        }
        if(nUs4OEMsSetting.has_value()) {
            nUs4OEMs = nUs4OEMsSetting.value();
        }
        else {
            // get number of us4oems from the probe adapter mapping
            // Determined based on ADAPTER MAPPINGS
            nUs4OEMs = getNumberOfModules(adapterMapping);
        }

        // Convert to list of us4oem mappings and active channel groups
        auto const nRx = Us4OEMImpl::N_RX_CHANNELS;
        auto const nTx = Us4OEMImpl::N_TX_CHANNELS;

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
        for(auto[module, channel] : adapterMapping) {
            // Channel mapping
            const auto group = channel / nRx;
            const auto element = currentRxGroupElement[module];
            if(element == 0) {
                // Starting new group
                currentRxGroup[module] = (ChannelIdx) group;
            } else {
                // Safety condition
                ARRUS_REQUIRES_TRUE(
                    group == currentRxGroup[module],
                    "Invalid probe adapter Rx channel mapping: inconsistent groups of channel "
                    "(consecutive elements of N_RX_CHANNELS are required)");
            }
            auto logicalChannel = group * nRx + element;
            us4oemChannelMapping[module][logicalChannel] = channel;
            adapterChannelMapping.emplace_back(module, ChannelIdx(logicalChannel));

            currentRxGroupElement[module] = (currentRxGroupElement[module] + 1) % nRx;
        }
        // END OF THE MASKS PRODUCTION
        for(int i = 0; i < nUs4OEMs; ++i) {
            result.push_back(Us4OEMSettings(us4oemChannelMapping[i], rxSettings, reprogrammingMode, txFrequencyRange));
        }
        return {result,
                ProbeAdapterSettings(
                    probeAdapterSettings.getModelId(),
                    probeAdapterSettings.getNumberOfChannels(),
                    adapterChannelMapping,
                    probeAdapterSettings.getIOSettings()
                )};
    }

private:
    static Ordinal getNumberOfModules(const ProbeAdapterSettings::ChannelMapping &adapterMapping) {
        std::vector<bool> mask(std::numeric_limits<Ordinal>::max());
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

    ProbeAdapterSettings::ChannelMapping remapUs4OEMs(const ProbeAdapterSettings::ChannelMapping &input,
                                                      const std::vector<Ordinal> &map) {
        ProbeAdapterSettings::ChannelMapping result(input.size());
        for(size_t i = 0; i < input.size(); ++i) {
            auto [module, channel] = input[i];
            result[i] = {map[module], channel};
        }
        return result;
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4SETTINGSCONVERTERIMPL_H
