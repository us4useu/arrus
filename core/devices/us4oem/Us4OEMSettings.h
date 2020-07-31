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
     * @param dtgcAttenuation
     * @param pgaGain
     * @param lnaGain
     * @param lpfCutoff
     * @param activeTermination
     * @param tgcSamples
     */
    Us4OEMSettings(
            std::vector<ChannelIdx> channelMapping,
            BitMask activeChannelGroups,
            const std::optional<uint8> dtgcAttenuation,
            const uint8 pgaGain,
            const uint8 lnaGain,
            const uint32 lpfCutoff,
            const std::optional<uint16> activeTermination,
            std::optional<TGCCurve> tgcSamples
    ) : channelMapping(std::move(channelMapping)),
        activeChannelGroups(std::move(activeChannelGroups)),
        dtgcAttenuation(dtgcAttenuation), pgaGain(pgaGain), lnaGain(lnaGain),
        lpfCutoff(lpfCutoff), activeTermination(activeTermination),
        tgcSamples(std::move(tgcSamples))
        {}

    [[nodiscard]] const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    [[nodiscard]] const BitMask &getActiveChannelGroups() const {
        return activeChannelGroups;
    }

    [[nodiscard]] std::optional<uint8> getDTGCAttenuation() const {
        return dtgcAttenuation;
    }

    [[nodiscard]] uint8 getPGAGain() const {
        return pgaGain;
    }

    [[nodiscard]] uint8 getLNAGain() const {
        return lnaGain;
    }

    [[nodiscard]] uint32 getLPFCutoff() const {
        return lpfCutoff;
    }

    [[nodiscard]] std::optional<uint16> getActiveTermination() const {
        return activeTermination;
    }

    [[nodiscard]] const std::optional<TGCCurve> &getTGCSamples() const {
        return tgcSamples;
    }

private:
    std::vector<ChannelIdx> channelMapping;
    BitMask activeChannelGroups;

    std::optional<uint8> dtgcAttenuation;
    uint8 pgaGain;
    uint8 lnaGain;

    uint32 lpfCutoff;
    std::optional<uint16> activeTermination;

    std::optional<TGCCurve> tgcSamples;
};

}

#endif //ARRUS_CORE_DEVICES_US4OEMSETTINGS_H
