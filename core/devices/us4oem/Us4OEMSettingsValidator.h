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
    :DeviceSettingsValidator<Us4OEMSettings>(
            DeviceId(DeviceType::Us4OEM, moduleOrdinal)){}

    void validate(const Us4OEMSettings &obj) override {
        // Active channel groups
        expectEqual<unsigned>(obj.getActiveChannelGroups().size(), 16,
                              "Number of active channel groups");

        // Channel mapping:
        // The size of the mapping:
        // Us4OEM mapping should include all channels, we don't want
        // the situation, where some o channels are
        expectEqual<unsigned>(obj.getChannelMapping().size(), 128,
                              "Size of channel mapping array");
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
        // TGC values
        if(obj.getDTGCAttenuation().has_value()) {
            expectOneOf(obj.getDTGCAttenuation().value(),
                        DTGCAttenuationValueMap::getInstance().getAvailableValues(),
                        "dtgc attenuation"
            );
        }
        expectOneOf(obj.getPGAGain(),
                    PGAGainValueMap::getInstance().getAvailableValues(),
                    "pga gain");
        expectOneOf(obj.getLNAGain(),
                    LNAGainValueMap::getInstance().getAvailableValues(),
                    "lna gain");

        if(!obj.getTGCSamples().empty()) {
            // Maximum/minimum number of samples.
            expectInRange<unsigned>(
                    obj.getTGCSamples().size(),
                    1, 1022,
                    "number of TGC samples");

            // Maximum/minimum value of a TGC sample.
            auto tgcMax = float(obj.getPGAGain() + obj.getLNAGain());
            auto tgcMin = float(tgcMax - 40);
            for(auto value : obj.getTGCSamples()) {
                expectInRange(value, tgcMin, tgcMax, "tgc sample");
            }
        }

        // Active termination.
        if(obj.getActiveTermination().has_value()) {
            expectOneOf(obj.getActiveTermination().value(),
                ActiveTerminationValueMap::getInstance().getAvailableValues(),
                "active termination");
        }

        // LPF cutoff.
        expectOneOf(obj.getLPFCutoff(),
                    LPFCutoffValueMap::getInstance().getAvailableValues(),
                    "lpf cutoff");
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
