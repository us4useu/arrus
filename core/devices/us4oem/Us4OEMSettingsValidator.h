#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H

#include <unordered_set>

#include "arrus/core/common/validation.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4oem/Us4OEMSettings.h"
#include "arrus/core/devices/DeviceSettingsValidator.h"

#include "arrus/core/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/external/ius4oem/ActiveTerminationValueMap.h"

namespace arrus {

class Us4OEMSettingsValidator : public DeviceSettingsValidator<Us4OEMSettings> {

public:

    explicit Us4OEMSettingsValidator(Ordinal moduleOrdinal)
            : DeviceSettingsValidator<Us4OEMSettings>(
            DeviceId(DeviceType::Us4OEM, moduleOrdinal)) {}

    void validate(const Us4OEMSettings &obj) override {
        // Active channel groups
        expectEqual<unsigned>("active channel groups",
                              obj.getActiveChannelGroups().size(), 16,
                              "(size)");

        // Channel mapping:
        // The size of the mapping:
        // Us4OEM mapping should include all channels, we don't want
        // the situation, where some o channels are
        expectEqual<unsigned>("channel mapping",
                              obj.getChannelMapping().size(), 128,
                              "(size)");
        if(obj.getChannelMapping().size() == 128) {
            auto &channelMapping = obj.getChannelMapping();
            // Check if contains (possibly permuted) groups:
            // 0-31, 32-63, 64-95, 96-127
            for(unsigned char group = 0; group < 4; ++group) {
                std::unordered_set<ChannelIdx> groupValues{
                        std::begin(channelMapping) + group * 32,
                        std::begin(channelMapping) + (group + 1) * 32};

                std::vector<ChannelIdx> missingValues;
                for(ChannelIdx j = group * 32;
                    j < (ChannelIdx) (group + 1) * 32; ++j) {
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
                                toString(missingValues),
                                group * 32, (group + 1) * 32
                        )
                );
            }
        }
        // TGC samples
        if(obj.getDTGCAttenuation().has_value()) {
            expectOneOf(
                    "dtgc attenuation",
                    obj.getDTGCAttenuation().value(),
                    DTGCAttenuationValueMap::getInstance().getAvailableValues()
            );
        }
        expectOneOf(
                "pga gain",
                obj.getPGAGain(),
                PGAGainValueMap::getInstance().getAvailableValues());
        expectOneOf(
                "lna gain",
                obj.getLNAGain(),
                LNAGainValueMap::getInstance().getAvailableValues());

        if(!obj.getTGCSamples().empty()) {
            // Maximum/minimum number of samples.
            expectInRange<unsigned>(
                    "tgc samples",
                    obj.getTGCSamples().size(),
                    1, 1022,
                    "(size)"
            );

            // Maximum/minimum value of a TGC sample.
            auto tgcMax = float(obj.getPGAGain() + obj.getLNAGain());
            auto tgcMin = std::max(0.0f, float(tgcMax - 40));
            expectAllInRange("tgc samples", obj.getTGCSamples(), tgcMin,
                             tgcMax);
        }

        // Active termination.
        if(obj.getActiveTermination().has_value()) {
            expectOneOf(
                    "active termination",
                    obj.getActiveTermination().value(),
                    ActiveTerminationValueMap::getInstance().getAvailableValues()
            );
        }

        // LPF cutoff.
        expectOneOf(
                "lpf cutoff",
                obj.getLPFCutoff(),
                LPFCutoffValueMap::getInstance().getAvailableValues()
        );
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
