#ifndef ARRUS_CORE_API_DEVICES_US4OEMSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4OEMSETTINGS_H

#include <utility>
#include <vector>
#include <bitset>
#include <optional>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/tgc.h"

namespace arrus {

/**
 * Us4OEM settings.
 *
 * Contains all raw parameters used to configure module.
 */
class Us4OEMSettings {
public:
    /**
     * Us4OEM Settings constructor.
     *
     * @param activeChannelGroups determines which groups of channels should be
     *        'active'. When the 'channel is active', Us4OEM can transmit/receive
     *        a signal through this channel.
     *        If the size of the group is equal `n`, and the number of module's
     *        channels is `m`, `activeChannelGroups[0]` turns on/off channels
     *        `0,1,..,(n-1)`, `activeChannelGroups[1]` turns on/off channels
     *        `n,(n+1),..,(2n-1)`, and so on. The value `m' is always divisible
     *        by `n`. The array `activeChannelGroups` should have exactly
     *        `m/n` elements.
     * @param channelMapping channel permutation to apply on a given Us4OEM.
     *  channelMapping[i] = j, where `i` is the virtual(logical) channel number,
     *  `j` is the physical channel number.
     * @param tgcSettings
     * @param lpfCutoff
     * @param activeTermination
     * @param tgcSamples tgc curve to apply, empty list means to turn off TGC
     */
    Us4OEMSettings(std::vector<ChannelIdx> channelMapping,
                   BitMask activeChannelGroups,
                   TGCSettings tgcSettings, uint32 lpfCutoff,
                   const std::optional<uint16> &activeTermination)
            : channelMapping(std::move(channelMapping)),
              activeChannelGroups(std::move(activeChannelGroups)),
              tgcSettings(std::move(tgcSettings)), lpfCutoff(lpfCutoff),
              activeTermination(activeTermination) {}

    [[nodiscard]] const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    [[nodiscard]] const BitMask &getActiveChannelGroups() const {
        return activeChannelGroups;
    }

    [[nodiscard]] const TGCSettings &getTGCSettings() const {
        return tgcSettings;
    }

    [[nodiscard]] uint32 getLPFCutoff() const {
        return lpfCutoff;
    }

    [[nodiscard]] const std::optional<uint16> &getActiveTermination() const {
        return activeTermination;
    }


private:
    std::vector<ChannelIdx> channelMapping;
    BitMask activeChannelGroups;

    TGCSettings tgcSettings;
    uint32 lpfCutoff;
    std::optional<uint16> activeTermination;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4OEMSETTINGS_H
