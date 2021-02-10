#ifndef ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
#define ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H

#include <utility>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

/**
 * Frame channel mapping: logical (frame, channel) -> physical (frame, channel)
 */
class FrameChannelMapping {
public:
    using Handle = std::unique_ptr<FrameChannelMapping>;
    using SharedHandle = std::shared_ptr<FrameChannelMapping>;
    using FrameNumber = uint16;
    constexpr static int8 UNAVAILABLE = -1;

    /**
     * Returns physical frame number and channel number for a given,
     * logical, frame number and a **rx aperture** channel.
     *
     * @param frame logical frame number
     * @param channel logical channel number
     * @return actual frame number and channel number
     */
     // TODO use FrameNumber typedef (current implementation is simplified for swig)
    virtual std::pair<unsigned short, arrus::int8> getLogical(FrameNumber frame, ChannelIdx channel) = 0;

    virtual FrameNumber getNumberOfLogicalFrames() = 0;
    virtual ChannelIdx getNumberOfLogicalChannels() = 0;

    /**
     * Returns true if the given PHYSICAL channel number is unavailable.
     *
     * @param channelNumber physical channel number to verify.
     * @return true if given channel is unavailable, false otherwise
     */
    static bool isChannelUnavailable(int8 channelNumber) {
        return channelNumber == UNAVAILABLE;
    }

    virtual ~FrameChannelMapping() = default;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
