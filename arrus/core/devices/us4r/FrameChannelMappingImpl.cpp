#include "FrameChannelMappingImpl.h"

#include <utility>
#include <iostream>

#include "arrus/common/asserts.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus::devices {

FrameChannelMappingImpl::FrameChannelMappingImpl(
        Us4OEMMapping &us4oemMapping, FrameMapping &frameMapping, ChannelMapping &channelMapping,
        std::vector<uint32> frameOffsets)
    : us4oemMapping(std::move(us4oemMapping)), frameMapping(std::move(frameMapping)),
    channelMapping(std::move(channelMapping)), frameOffsets(std::move(frameOffsets)) {

    ARRUS_REQUIRES_TRUE_E(frameMapping.rows() == channelMapping.rows()
                          && frameMapping.cols() == channelMapping.cols()
                          && frameMapping.rows() == us4oemMapping.rows()
                          && frameMapping.cols() == us4oemMapping.cols(),
                          ArrusException("All channel mapping structures should have the same shape"));
}

FrameChannelMappingImpl::~FrameChannelMappingImpl() = default;

FrameChannelMappingAddress
FrameChannelMappingImpl::getLogical(FrameNumber frame, ChannelIdx channel) const {
    auto us4oem = us4oemMapping(frame, channel);
    auto physicalFrame = frameMapping(frame, channel);
    auto physicalChannel = channelMapping(frame, channel);
    return FrameChannelMappingAddress{us4oem, physicalFrame, physicalChannel};
}

FrameChannelMapping::FrameNumber FrameChannelMappingImpl::getNumberOfLogicalFrames() const {
    ARRUS_REQUIRES_TRUE(frameMapping.rows() >= 0 && frameMapping.rows() <= std::numeric_limits<uint16>::max(),
                        "FCM number of logical frames exceeds the maximum number of frames (uint16::max).");
    return static_cast<FrameChannelMapping::FrameNumber>(frameMapping.rows());
}

ChannelIdx FrameChannelMappingImpl::getNumberOfLogicalChannels() const {
    ARRUS_REQUIRES_TRUE(frameMapping.cols() >= 0 && frameMapping.cols() <= std::numeric_limits<uint16>::max(),
                        "FCM number of logical channels exceeds the maximum number of channels (uint16::max).");
    return static_cast<ChannelIdx>(frameMapping.cols());
}

uint32 FrameChannelMappingImpl::getFirstFrame(uint8 us4oem) const {
    return frameOffsets[us4oem];
}

const std::vector<uint32> & FrameChannelMappingImpl::getFrameOffsets() const {
    return frameOffsets;
}

void FrameChannelMappingBuilder::setChannelMapping(FrameNumber logicalFrame, ChannelIdx logicalChannel,
                                                   uint8 us4oem, FrameNumber physicalFrame, int8 physicalChannel) {
    us4oemMapping(logicalFrame, logicalChannel) = us4oem;
    frameMapping(logicalFrame, logicalChannel) = physicalFrame;
    channelMapping(logicalFrame, logicalChannel) = physicalChannel;
}

FrameChannelMappingImpl::Handle FrameChannelMappingBuilder::build() {
    return std::make_unique<FrameChannelMappingImpl>(this->us4oemMapping, this->frameMapping, this->channelMapping,
                                                     this->frameOffsets);
}

FrameChannelMappingBuilder::FrameChannelMappingBuilder(FrameNumber nFrames, ChannelIdx nChannels)
    : us4oemMapping(FrameChannelMappingImpl::Us4OEMMapping(nFrames, nChannels)),
      frameMapping(FrameChannelMappingImpl::FrameMapping(nFrames, nChannels)),
      channelMapping(FrameChannelMappingImpl::ChannelMapping(nFrames, nChannels)) {
    // Creates empty frame mapping.
    us4oemMapping.fill(0);
    frameMapping.fill(0);
    channelMapping.fill(FrameChannelMapping::UNAVAILABLE);
}

void FrameChannelMappingBuilder::setFrameOffsets(const std::vector<uint32> &offsets) {
    this->frameOffsets = offsets;
}

}

