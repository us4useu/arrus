#ifndef ARRUS_CORE_DEVICES_US4OEM_TESTS_COMMONS_H
#define ARRUS_CORE_DEVICES_US4OEM_TESTS_COMMONS_H

#include "arrus/core/common/collections.h"
#include "arrus/core/api/devices/us4oem/Us4OEMSettings.h"

using namespace arrus;

struct TestUs4OEMSettings {
    std::vector<ChannelIdx> channelMapping{getRange<ChannelIdx>(0, 128)};
    BitMask activeChannelGroups{getNTimes<bool>(true, 16)};
    std::optional<uint16> dtgcAttenuation{42};
    uint16 pgaGain{30};
    uint16 lnaGain{24};
    TGCCurve tgcSamples{getRange<float>(30, 40, 0.5)};
    uint32 lpfCutoff{(int) 10e6};
    std::optional<uint16> activeTermination{50};

    std::vector<std::string> invalidParameters;

    Us4OEMSettings getUs4OEMSettings() const {
        return Us4OEMSettings(channelMapping, activeChannelGroups,
                              dtgcAttenuation, pgaGain,
                              lnaGain, lpfCutoff, activeTermination,
                              tgcSamples);
    }

    friend std::ostream &
    operator<<(std::ostream &os, const TestUs4OEMSettings &settings) {
        os << "channelMapping: " << toString(settings.channelMapping)
           << " activeChannelGroups: " << toString(settings.activeChannelGroups)
           << " dtgcAttenuation: " << toString(settings.dtgcAttenuation)
           << " pgaGain: " << (int) settings.pgaGain
           << " lnaGain: " << (int) settings.lnaGain
           << " lpfCutoff: " << settings.lpfCutoff
           << " activeTermination: " << toString(settings.activeTermination)
           << " tgcSamples: " << toString(settings.tgcSamples);

        for(const auto& invalidParameter : settings.invalidParameters) {
            os << " invalidParameter: " << invalidParameter;
        }
        return os;
    }
};


#endif //ARRUS_CORE_DEVICES_US4OEM_TESTS_COMMONS_H
