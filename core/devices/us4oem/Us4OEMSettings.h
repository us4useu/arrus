#ifndef ARRUS_CORE_DEVICES_US4OEMSETTINGS_H
#define ARRUS_CORE_DEVICES_US4OEMSETTINGS_H

#include <vector>
#include <bitset>
#include <optional>

#include "core/types.h"

namespace arrus {

/**
 * Us4OEM settings.
 *
 * Contains all raw parameters used to configure module.
 */
class Us4OEMSettings {
public:
    Us4OEMSettings(
            const std::vector<ChannelIdx> &channelMapping,
            const BitMask activeChannelGroups,
            const std::optional<double> dtgc,
            const double pgaGain,
            const double lnaGain,
            const double lpfCutoff,
            const double activeTermination,
            const std::optional<TGCCurve> &tgcSamples
    ) : channelMapping(channelMapping),
        activeChannelGroups(activeChannelGroups),
        dtgc(dtgc), pgaGain(pgaGain), lnaGain(lnaGain),
        lpfCutoff(lpfCutoff), activeTermination(activeTermination),
        tgcSamples(tgcSamples)
        {}

    const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    const BitMask &getActiveChannelGroups() const {
        return activeChannelGroups;
    }

    std::optional<double> getDTGC() const {
        return dtgc;
    }

    double getPGAGain() const {
        return pgaGain;
    }

    double getLNAGain() const {
        return lnaGain;
    }

    double getLPFCutoff() const {
        return lpfCutoff;
    }

    double getActiveTermination() const {
        return activeTermination;
    }

    const std::optional<TGCCurve> &getTGCSamples() const {
        return tgcSamples;
    }

private:
    std::vector<ChannelIdx> channelMapping;
    BitMask activeChannelGroups;
    std::optional<double> dtgc;
    double pgaGain;
    double lnaGain;
    double lpfCutoff;
    double activeTermination;

    std::optional<TGCCurve> tgcSamples;
};

}

#endif //ARRUS_CORE_DEVICES_US4OEMSETTINGS_H
