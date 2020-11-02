#ifndef ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
#define ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H

#include <utility>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

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
     // TODO use FrameNumber typedef (simplified current implementation for swig)
    virtual std::pair<unsigned short, arrus::int8> getLogical(FrameNumber frame, ChannelIdx channel) = 0;

    virtual FrameNumber getNumberOfLogicalFrames() = 0;
    virtual ChannelIdx getNumberOfLogicalChannels() = 0;

    static bool isChannelUnavailable(int8 channelNumber) {
        return channelNumber == UNAVAILABLE;
    }

    virtual ~FrameChannelMapping() = default;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
