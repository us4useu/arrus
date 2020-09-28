#include "FrameChannelMappingImpl.h"

#include <utility>

#include "arrus/common/asserts.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus::devices {

FrameChannelMappingImpl::FrameChannelMappingImpl(FrameMapping &frameMapping,
                                                 ChannelMapping &channelMapping)
    : frameMapping(std::move(frameMapping)), channelMapping(std::move(channelMapping)) {

    ARRUS_REQUIRES_TRUE_E(frameMapping.rows() == channelMapping.rows()
                          && frameMapping.cols() == channelMapping.cols(),
                          ArrusException("Frame and channel mapping arrays should have the "
                                         "same shape"));
}

std::pair<FrameChannelMapping::FrameNumber, int8>
FrameChannelMappingImpl::getChannel(FrameNumber frame, ChannelIdx channel) {
    auto physicalFrame = frameMapping(frame, channel);
    auto physicalChannel = channelMapping(frame, channel);
    return {physicalFrame, physicalChannel};
}

uint32 FrameChannelMappingImpl::getNumberOfFrames() {
    return frameMapping.rows();
}

void
FrameChannelMappingBuilder::setChannelMapping(FrameNumber logicalFrame, ChannelIdx logicalChannel,
                                              FrameNumber physicalFrame, int8 physicalChannel) {
    frameMapping(logicalFrame, logicalChannel) = physicalFrame;
    channelMapping(logicalFrame, logicalChannel) = physicalChannel;
}

FrameChannelMappingImpl::Handle FrameChannelMappingBuilder::build() {
    return std::make_unique<FrameChannelMappingImpl>(this->frameMapping, this->channelMapping);
}

FrameChannelMappingBuilder::FrameChannelMappingBuilder(FrameNumber nFrames, ChannelIdx nChannels)
    : frameMapping(FrameChannelMappingImpl::FrameMapping(nFrames, nChannels)),
      channelMapping(FrameChannelMappingImpl::ChannelMapping(nFrames, nChannels)) {

    frameMapping.fill(FrameChannelMapping::UNAVAILABLE);
    channelMapping.fill(FrameChannelMapping::UNAVAILABLE);
}

}

