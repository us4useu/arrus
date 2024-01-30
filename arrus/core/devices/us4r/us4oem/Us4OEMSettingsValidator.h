#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMSETTINGSVALIDATOR_H

#include <unordered_set>

#include "arrus/core/common/validation.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/SettingsValidator.h"
#include "arrus/core/devices/us4r/validators/RxSettingsValidator.h"

#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/ActiveTerminationValueMap.h"

namespace arrus::devices {

class Us4OEMSettingsValidator : public SettingsValidator<Us4OEMSettings> {
public:
    explicit Us4OEMSettingsValidator(Ordinal moduleOrdinal)
            : SettingsValidator<Us4OEMSettings>(DeviceId(DeviceType::Us4OEM, moduleOrdinal)) {}

    void validate(const Us4OEMSettings &obj) {
        constexpr ChannelIdx RX_SIZE = Us4OEMImpl::N_RX_CHANNELS;
        constexpr ChannelIdx N_TX_CHANNELS = Us4OEMImpl::N_TX_CHANNELS;
        constexpr ChannelIdx N_RX_GROUPS = N_TX_CHANNELS / RX_SIZE;

        // Channel mapping:
        // The size of the mapping:
        // Us4OEM mapping should include all channels, we don't want
        // the situation, where some o channels are not defined.
        expectEqual("channel mapping", obj.getChannelMapping().size(), (size_t) N_TX_CHANNELS, "(size)");

        if(obj.getChannelMapping().size() == (size_t) N_TX_CHANNELS) {
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
                expectTrue("channel mapping", missingValues.empty(),
                           arrus::format("Some of Us4OEM channels: '{}' are missing in the group of channels [{}, {}]",
                                         ::arrus::toString(missingValues), group * RX_SIZE, (group + 1) * RX_SIZE));
            }
        }
        RxSettingsValidator rxSettingsValidator;
        rxSettingsValidator.validate(obj.getRxSettings());
        copyErrorsFrom(rxSettingsValidator);
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMSETTINGSVALIDATOR_H
