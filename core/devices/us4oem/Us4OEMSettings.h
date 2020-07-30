#ifndef ARRUS_CORE_DEVICES_US4OEMSETTINGS_H
#define ARRUS_CORE_DEVICES_US4OEMSETTINGS_H

#include <utility>
#include <vector>
#include <bitset>
#include <optional>

#include "core/common/types.h"

namespace arrus {

/**
 * Us4OEM settings.
 *
 * Contains all raw parameters used to configure module.
 */
class Us4OEMSettings {
public:
    /**
     * Creates new Us4OEM configuration.
     *
     * @param channelMapping
     * @param activeChannelGroups
     * @param dtgc
     * @param pgaGain
     * @param lnaGain
     * @param lpfCutoff
     * @param activeTermination
     * @param tgcSamples
     */
    Us4OEMSettings(
            std::vector<ChannelIdx> channelMapping,
            BitMask activeChannelGroups,
            const std::optional<double> dtgc,
            const double pgaGain,
            const double lnaGain,
            const double lpfCutoff,
            const double activeTermination,
            std::optional<TGCCurve> tgcSamples
    ) : channelMapping(std::move(channelMapping)),
        activeChannelGroups(std::move(activeChannelGroups)),
        dtgc(dtgc), pgaGain(pgaGain), lnaGain(lnaGain),
        lpfCutoff(lpfCutoff), activeTermination(activeTermination),
        tgcSamples(std::move(tgcSamples))
        {}

    [[nodiscard]] const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    [[nodiscard]] const BitMask &getActiveChannelGroups() const {
        return activeChannelGroups;
    }

    [[nodiscard]] std::optional<double> getDTGC() const {
        return dtgc;
    }

    [[nodiscard]] double getPGAGain() const {
        return pgaGain;
    }

    [[nodiscard]] double getLNAGain() const {
        return lnaGain;
    }

    [[nodiscard]] double getLPFCutoff() const {
        return lpfCutoff;
    }

    [[nodiscard]] double getActiveTermination() const {
        return activeTermination;
    }

    [[nodiscard]] const std::optional<TGCCurve> &getTGCSamples() const {
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
