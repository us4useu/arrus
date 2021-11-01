#ifndef ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H
#define ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H

#include <vector>
#include <utility>
#include <Eigen/Dense>


#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"


namespace arrus::devices {

class FrameChannelMappingImpl : public FrameChannelMapping {
public:
    using Handle = std::unique_ptr<FrameChannelMappingImpl>;
    using Us4OEMMapping = Eigen::Matrix<Us4OEMNumber, Eigen::Dynamic, Eigen::Dynamic>;
    using FrameMapping = Eigen::Matrix<FrameNumber, Eigen::Dynamic, Eigen::Dynamic>;
    using ChannelMapping = Eigen::Matrix<int8, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * Takes ownership for the provided frames.
     */
    FrameChannelMappingImpl(Us4OEMMapping &us4oemMapping, FrameMapping &frameMapping, ChannelMapping &channelMapping);

    std::tuple<Us4OEMNumber, FrameNumber, int8> getLogical(FrameNumber frame, ChannelIdx channel) override;

    FrameNumber getNumberOfLogicalFrames() override;

    ChannelIdx getNumberOfLogicalChannels() override;

    ~FrameChannelMappingImpl() override;

private:
    // logical (frame, number) -> physical (us4oem, frame, number)
    Us4OEMMapping us4oemMapping;
    FrameMapping frameMapping;
    ChannelMapping channelMapping;
};

class FrameChannelMappingBuilder {
public:
    using FrameNumber = FrameChannelMapping::FrameNumber;
    using Us4OEMNumber = FrameChannelMapping::Us4OEMNumber;

    FrameChannelMappingBuilder(FrameNumber nFrames, ChannelIdx nChannels);

    void setChannelMapping(FrameNumber logicalFrame, ChannelIdx logicalChannel, // ->
                           Us4OEMNumber us4oem, FrameNumber physicalFrame, int8 physicalChannel);

    FrameChannelMappingImpl::Handle build();

private:
    // logical (frame, number) -> physical (frame, number)
    FrameChannelMappingImpl::Us4OEMMapping us4oemMapping;
    FrameChannelMappingImpl::FrameMapping frameMapping;
    FrameChannelMappingImpl::ChannelMapping channelMapping;
};

}


#endif //ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H
