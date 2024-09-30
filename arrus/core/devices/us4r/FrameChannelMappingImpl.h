#ifndef ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H
#define ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H

#include <Eigen/Dense>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"

// Make the FrameChannelMappingAddress available for structure binding
namespace std {

template<> struct tuple_size<::arrus::devices::FrameChannelMappingAddress> : integral_constant<size_t, 3> {};

template<> struct tuple_element<0, ::arrus::devices::FrameChannelMappingAddress> {
    using type = ::arrus::devices::FrameChannelMapping::Us4OEMNumber;
};

template<> struct tuple_element<1, ::arrus::devices::FrameChannelMappingAddress> {
    using type = ::arrus::devices::FrameChannelMapping::FrameNumber;
};

template<> struct tuple_element<2, ::arrus::devices::FrameChannelMappingAddress> {
    using type = int8_t;
};
}// namespace std

namespace arrus::devices {
template<std::size_t Index>
std::tuple_element_t<Index, ::arrus::devices::FrameChannelMappingAddress>
get(const ::arrus::devices::FrameChannelMappingAddress &address) {
    static_assert(Index < 3, "Index out of bounds for FrameChannelMappingAddress");
    if constexpr (Index == 0)
        return address.getUs4oem();
    if constexpr (Index == 1)
        return address.getFrame();
    if constexpr (Index == 2)
        return address.getChannel();
}

}// namespace arrus::devices

namespace arrus::devices {

class FrameChannelMappingBuilder;

class FrameChannelMappingImpl : public FrameChannelMapping {
public:
    using Handle = std::unique_ptr<FrameChannelMappingImpl>;
    using Us4OEMMapping = Eigen::Matrix<Us4OEMNumber, Eigen::Dynamic, Eigen::Dynamic>;
    using FrameMapping = Eigen::Matrix<FrameNumber, Eigen::Dynamic, Eigen::Dynamic>;
    using ChannelMapping = Eigen::Matrix<int8, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * Takes ownership for the provided frames.
     */
    FrameChannelMappingImpl(Us4OEMMapping us4oemMapping, FrameMapping frameMapping, ChannelMapping channelMapping,
                            std::vector<uint32> frameOffsets = {0}, std::vector<uint32> numberOfFrames = {0});

    FrameChannelMappingAddress getLogical(FrameNumber frame, ChannelIdx channel) const override;

    uint32 getFirstFrame(uint8 us4oem) const override;

    uint32 getNumberOfFrames(uint8 us4oem) const override;

    const std::vector<uint32> &getFrameOffsets() const override;

    const std::vector<uint32> &getNumberOfFrames() const override;

    FrameNumber getNumberOfLogicalFrames() const override;

    ChannelIdx getNumberOfLogicalChannels() const override;

    ~FrameChannelMappingImpl() override;

    std::string toString() const override;

private:
    friend class FrameChannelMappingBuilder;
    // logical (frame, number) -> physical (us4oem, frame, number)
    Us4OEMMapping us4oemMapping;
    FrameMapping frameMapping;
    ChannelMapping channelMapping;
    std::vector<uint32> frameOffsets;
    std::vector<uint32> numberOfFrames;
};

class FrameChannelMappingBuilder {
public:
    using FrameNumber = FrameChannelMapping::FrameNumber;
    using Us4OEMNumber = FrameChannelMapping::Us4OEMNumber;

    static FrameChannelMappingBuilder like(const FrameChannelMapping &mapping) {
        FrameChannelMappingBuilder builder{mapping.getNumberOfLogicalFrames(),
                                           mapping.getNumberOfLogicalChannels()};
        builder.setFrameOffsets(mapping.getFrameOffsets());
        builder.setNumberOfFrames(mapping.getNumberOfFrames());
        return builder;
    }

    static FrameChannelMappingBuilder copy(const FrameChannelMappingImpl &mapping) {
        FrameChannelMappingBuilder builder;
        // Copy
        builder.us4oemMapping = mapping.us4oemMapping;
        builder.frameMapping = mapping.frameMapping;
        builder.channelMapping = mapping.channelMapping;
        builder.setFrameOffsets(mapping.getFrameOffsets());
        builder.setNumberOfFrames(mapping.getNumberOfFrames());
        return builder;
    }

    FrameChannelMappingBuilder(FrameNumber nFrames, ChannelIdx nChannels);

    void setChannelMapping(FrameNumber logicalFrame, ChannelIdx logicalChannel, Us4OEMNumber us4oem,
                           FrameNumber physicalFrame, int8 physicalChannel);

    FrameChannelMappingImpl::Handle build();
    void setFrameOffsets(const std::vector<uint32> &frameOffsets);
    void setNumberOfFrames(const std::vector<uint32> &nFrames);
    void slice(FrameNumber start, FrameNumber end);
    void subtractPhysicalFrameNumber(Ordinal oem, FrameNumber offset);
    void recalculateOffsets();

private:
    // logical (frame, number) -> physical (frame, number)
    FrameChannelMappingBuilder() = default;
    FrameChannelMappingImpl::Us4OEMMapping us4oemMapping;
    FrameChannelMappingImpl::FrameMapping frameMapping;
    FrameChannelMappingImpl::ChannelMapping channelMapping;
    std::vector<uint32> frameOffsets = {0};
    std::vector<uint32> numberOfFrames = {0};
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_FRAMECHANNELMAPPINGIMPL_H
