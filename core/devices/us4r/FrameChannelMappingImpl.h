#ifndef ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H
#define ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H

#include <vector>
#include <utility>
#include <Eigen/Dense>

#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"


namespace arrus::devices {

class FrameChannelMappingImpl : public FrameChannelMapping {
public:
    using Handle = std::unique_ptr<FrameChannelMappingImpl>;
    using FrameMapping = Eigen::Matrix<FrameNumber, Eigen::Dynamic, Eigen::Dynamic>;
    using ChannelMapping = Eigen::Matrix<int8, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * Takes ownership for the provided frames.
     */
    FrameChannelMappingImpl(FrameMapping &frameMapping, ChannelMapping &channelMapping);

    /**
     * @param frame logical frame to acquire
     * @param channel channel in the logical frame to acquire
     * @return frame and channel number of the physical signal data (the one returned by us4r device)
     */
    std::pair<FrameNumber, int8> getChannel(FrameNumber frame, ChannelIdx channel) override;

    uint32 getNumberOfFrames() override;

private:
    // logical (frame, number) -> physical (frame, number)
    FrameMapping frameMapping;
    ChannelMapping channelMapping;
};

class FrameChannelMappingBuilder {
public:
    using FrameNumber = FrameChannelMapping::FrameNumber;

    FrameChannelMappingBuilder(FrameNumber nFrames, ChannelIdx nChannels);

    void setChannelMapping(FrameNumber logicalFrame, ChannelIdx logicalChannel,
                           FrameNumber physicalFrame, int8 physicalChannel);

    FrameChannelMappingImpl::Handle build();

private:
    // logical (frame, number) -> physical (frame, number)
    FrameChannelMappingImpl::FrameMapping frameMapping;
    FrameChannelMappingImpl::ChannelMapping channelMapping;
};

}


#endif //ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H
