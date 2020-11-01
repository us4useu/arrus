#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMSETTINGSVALIDATOR_H

#include <unordered_set>

#include "arrus/core/common/validation.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/SettingsValidator.h"

#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/ActiveTerminationValueMap.h"

namespace arrus::devices {

class Us4OEMSettingsValidator : public SettingsValidator<Us4OEMSettings> {
public:
    explicit Us4OEMSettingsValidator(Ordinal moduleOrdinal)
            : SettingsValidator<Us4OEMSettings>(
            DeviceId(DeviceType::Us4OEM, moduleOrdinal)) {}

    void validate(const Us4OEMSettings &obj) override {
        constexpr ChannelIdx RX_SIZE = Us4OEMImpl::N_RX_CHANNELS;
        constexpr ChannelIdx N_TX_CHANNELS = Us4OEMImpl::N_TX_CHANNELS;
        constexpr ChannelIdx N_RX_GROUPS = N_TX_CHANNELS / RX_SIZE;

        // Active channel groups
        expectEqual("active channel groups",
                              obj.getActiveChannelGroups().size(),
					(size_t)Us4OEMImpl::N_ACTIVE_CHANNEL_GROUPS,
                              "(size)");

        // Channel mapping:
        // The size of the mapping:
        // Us4OEM mapping should include all channels, we don't want
        // the situation, where some o channels are not defined.
        expectEqual("channel mapping",
                              obj.getChannelMapping().size(),
							  (size_t)N_TX_CHANNELS,
                              "(size)");

        if(obj.getChannelMapping().size() == (size_t)N_TX_CHANNELS) {
            auto &channelMapping = obj.getChannelMapping();

            // Check if contains (possibly permuted) groups:
            // 0-31, 32-63, 64-95, 96-127
            for(unsigned char group = 0; group < N_RX_GROUPS; ++group) {
                std::unordered_set<ChannelIdx> groupValues{
                        std::begin(channelMapping) + group * RX_SIZE,
                        std::begin(channelMapping) + (group + 1) * RX_SIZE};

                std::vector<ChannelIdx> missingValues;
                for(ChannelIdx j = group * RX_SIZE;
                    j < (ChannelIdx) (group + 1) * RX_SIZE; ++j) {
                    if(groupValues.find(j) == groupValues.end()) {
                        missingValues.push_back(j);
                    }
                }
                expectTrue(
                        "channel mapping",
                        missingValues.empty(),
                        arrus::format(
                                "Some of Us4OEM channels: '{}' "
                                "are missing in the group of channels [{}, {}]",
                                ::arrus::toString(missingValues),
                                group * RX_SIZE, (group + 1) * RX_SIZE
                        )
                );
            }
        }
        // TGC samples
        if(obj.getRxSettings().getDtgcAttenuation().has_value()) {
            expectOneOf(
                    "dtgc attenuation",
                    obj.getRxSettings().getDtgcAttenuation().value(),
                    DTGCAttenuationValueMap::getInstance().getAvailableValues()
            );
        }
        expectOneOf(
                "pga gain",
                obj.getRxSettings().getPgaGain(),
                PGAGainValueMap::getInstance().getAvailableValues());
        expectOneOf(
                "lna gain",
                obj.getRxSettings().getLnaGain(),
                LNAGainValueMap::getInstance().getAvailableValues());

        if(!obj.getRxSettings().getTgcSamples().empty()) {
            // Maximum/minimum number of samples.
            expectInRange(
                    "tgc samples",
                    obj.getRxSettings().getTgcSamples().size(),
					(size_t)1, (size_t)Us4OEMImpl::TGC_N_SAMPLES,
                    "(size)"
            );

            // Maximum/minimum value of a TGC sample.
            auto tgcMax = float(obj.getRxSettings().getPgaGain()
                                + obj.getRxSettings().getLnaGain());
            auto tgcMin = std::max(0.0f, float(tgcMax - Us4OEMImpl::TGC_ATTENUATION_RANGE));
            expectAllInRange("tgc samples",
                             obj.getRxSettings().getTgcSamples(), tgcMin,
                             tgcMax);
        }

        // Active termination.
        if(obj.getRxSettings().getActiveTermination().has_value()) {
            expectOneOf(
                    "active termination",
                    obj.getRxSettings().getActiveTermination().value(),
                    ActiveTerminationValueMap::getInstance().getAvailableValues()
            );
        }

        // LPF cutoff.
        expectOneOf(
                "lpf cutoff",
                obj.getRxSettings().getLpfCutoff(),
                LPFCutoffValueMap::getInstance().getAvailableValues()
        );
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMSETTINGSVALIDATOR_H
