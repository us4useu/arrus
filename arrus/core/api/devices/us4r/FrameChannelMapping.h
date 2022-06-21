#ifndef ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H
#define ARRUS_CORE_API_DEVICES_US4R_FRAMECHANNELMAPPING_H

#include <utility>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {


/**
 * A tuple that describes position of a given channel in a sequence
 * of frames produced by us4OEM modules.
 */
class FrameChannelMappingAddress {

public:
    FrameChannelMappingAddress(uint8 us4oem, unsigned short frame, int8 channel)
    : us4oem(us4oem), frame(frame), channel(channel) {}

    uint8 getUs4oem() const {
        return us4oem;
    }

    unsigned short getFrame() const {
        return frame;
    }

    int8 getChannel() const {
        return channel;
    }

    bool operator==(const FrameChannelMappingAddress &rhs) const {
        return us4oem == rhs.us4oem &&
               frame == rhs.frame &&
               channel == rhs.channel;
    }

    bool operator!=(const FrameChannelMappingAddress &rhs) const {
        return !(rhs == *this);
    }

private:
    arrus::uint8 us4oem;
    unsigned short frame;
    arrus::int8 channel;
};

/**
 * Frame channel mapping: logical (frame, channel) -> physical (frame, channel)
 */
class FrameChannelMapping {
public:
    using Handle = std::unique_ptr<FrameChannelMapping>;
    using SharedHandle = std::shared_ptr<FrameChannelMapping>;
    // Frame Channel Mapping supports up to 256 Us4OEMs.
    using Us4OEMNumber = uint8;
    using FrameNumber = uint16;
    constexpr static int8 UNAVAILABLE = -1;


    /**
     * Returns us4oem module number, physical frame number and channel number for a given,
     * logical, frame number and an **rx aperture** channel.
     *
     * @param frame logical frame number
     * @param channel logical channel number
     * @return a tuple: us4oem module number, frame number (within a single sequence), channel number
     */
    virtual FrameChannelMappingAddress getLogical(FrameNumber frame, ChannelIdx channel) const = 0;

    /**
     * Returns the number of frame where the given us4OEM data starts.
     * The frame number is computed taking into account the batch size and the number of frames in the
     * sequence of data produced by preceding us4OEM modules. That is, assuming the same number of samples
     * is acquired in each RF frame, you can get the address where us4oem data starts using the following
     * formula: the frame number * number of samples * 32 (number of us4OEM RX channels).
     *
     * @param us4oem us4oem ordinal number (0, 1, ...)
     * @return the number of frame, which starts portion of data acquired by the given us4OEM.
     */
    virtual arrus::uint32 getFirstFrame(arrus::uint8 us4oem) const = 0;

    /**
     * Returns the list of frame offsets ('position of first us4oem frame'). See `getFirstFrame` for more information.
     */
    virtual const std::vector<uint32> & getFrameOffsets() const = 0;

    virtual FrameNumber getNumberOfLogicalFrames() const = 0;
    virtual ChannelIdx getNumberOfLogicalChannels() const = 0;

    /**
     * Returns the number of frames that this us4OEM will transfer to the host device.
     *
     * @param us4oem us4OEM ordinal number.
     */
    virtual uint32 getNumberOfFrames(uint8 us4oem) const = 0;

    virtual const std::vector<uint32> &getNumberOfFrames() const = 0;

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
