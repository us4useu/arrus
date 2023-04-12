#ifndef ARRUS_CORE_API_DEVICES_US4R_US4OEMSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4OEMSETTINGS_H

#include <utility>
#include <vector>
#include <bitset>
#include <optional>
#include <ostream>
#include <unordered_set>

#include "arrus/core/api/common/types.h"
#include "RxSettings.h"

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
     * Determines when the us4OEM FPGA reprogramming starts.
     */
    enum class ReprogrammingMode {
        /* Us4OEM FPGA reprogramming starts after signal data acquisition is
         * ended. Total TX/RX time: rx time + reprogramming time.
         * Total TX/RX time determines possible maximum PRF.
         * This mode minimizes signal noise at the expense of additional
         * reprogramming time (which decreases available PRF). */
        SEQUENTIAL = 0,
        /* Us4OEM FPGA reprogramming for the next TX starts when the the current
         * TX is triggered; both processes (reprogramming for the next TX/RX
         * and current TX/RX) are done in parallel. Total TX/RX time:
         * max(rx time, reprogramming time).
         * Total TX/RX time determines possible maximum PRF.
         * This mode maximizes the possible PRF at the expense of additional
         * noise that may appear at the beginning of the data (emitted during
         * the FPGA reprogramming). */
        PARALLEL = 1
    };

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
     * @param reprogrammingMode us4OEM reprogramming mode
     * @param txFrequencyRange tx frequency range, actually: tx frequency divider, by default 1 is used.
     */
    Us4OEMSettings(ChannelMapping channelMapping,
                   BitMask activeChannelGroups,
                   RxSettings rxSettings,
                   std::unordered_set<uint8> channelsMask,
                   ReprogrammingMode reprogrammingMode = ReprogrammingMode::SEQUENTIAL,
                   int txFrequencyRange = 1)
            : channelMapping(std::move(channelMapping)),
              activeChannelGroups(std::move(activeChannelGroups)),
              rxSettings(std::move(rxSettings)),
              channelsMask(std::move(channelsMask)),
              reprogrammingMode(reprogrammingMode),
              txFrequencyRange(txFrequencyRange)
              {}


    const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    const BitMask &getActiveChannelGroups() const {
        return activeChannelGroups;
    }

    const RxSettings &getRxSettings() const {
        return rxSettings;
    }

    const std::unordered_set<uint8> &getChannelsMask() const {
        return channelsMask;
    }

    ReprogrammingMode getReprogrammingMode() const {
        return reprogrammingMode;
    }

    int getTxFrequencyRange() const {
        return txFrequencyRange;
    }

private:
    std::vector<ChannelIdx> channelMapping;
    BitMask activeChannelGroups;
    RxSettings rxSettings;
    std::unordered_set<uint8> channelsMask;
    ReprogrammingMode reprogrammingMode;
    int txFrequencyRange = 1;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4OEMSETTINGS_H
