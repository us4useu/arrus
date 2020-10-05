#ifndef ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
#define ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H

#include <utility>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

class FrameChannelMapping {
public:
    using Handle = std::unique_ptr<FrameChannelMapping>;
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
    virtual std::pair<FrameNumber, int8> getLogical(FrameNumber frame, ChannelIdx channel) = 0;

    virtual uint32 getNumberOfLogicalFrames() = 0;
    virtual uint32 getNumberOfLogicalChannels() = 0;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
