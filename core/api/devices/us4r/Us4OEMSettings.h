#ifndef ARRUS_CORE_API_DEVICES_US4R_US4OEMSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4OEMSETTINGS_H

#include <utility>
#include <vector>
#include <bitset>
#include <optional>
#include <ostream>
#include <unordered_set>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::devices {

/**
 * Us4OEM settings.
 *
 * Contains all raw parameters used to configure module.
 */
class Us4OEMSettings {
public:
    using ChannelMapping = std::vector<ChannelIdx>;

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
     * @param rxSettings initial rx settings to apply
     * @param channelMask channels that should be always turned off,
     *   CHANNEL NUMBERS STARTS FROM 0
     */
    Us4OEMSettings(ChannelMapping channelMapping,
                   BitMask activeChannelGroups,
                   RxSettings rxSettings,
                   std::unordered_set<uint8> channelsMask)
            : channelMapping(std::move(channelMapping)),
              activeChannelGroups(std::move(activeChannelGroups)),
              rxSettings(std::move(rxSettings)),
              channelsMask(std::move(channelsMask)){}


    [[nodiscard]] const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    [[nodiscard]] const BitMask &getActiveChannelGroups() const {
        return activeChannelGroups;
    }

    [[nodiscard]] const RxSettings &getRxSettings() const {
        return rxSettings;
    }

    [[nodiscard]] const std::unordered_set<uint8> &getChannelsMask() const {
        return channelsMask;
    }

private:
    std::vector<ChannelIdx> channelMapping;
    BitMask activeChannelGroups;
    RxSettings rxSettings;
    std::unordered_set<uint8> channelsMask;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4OEMSETTINGS_H
