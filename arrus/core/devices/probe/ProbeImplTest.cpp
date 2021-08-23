#include <gtest/gtest.h>
#include <ostream>
#include "arrus/core/devices/probe/ProbeImpl.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace {

using namespace arrus;
using namespace arrus::devices;

class ProbeImplFcmRemapTest : public ::testing::Test {
protected:
    void SetUp() override {

        FrameChannelMappingBuilder fcmBuilder(N_FRAMES, N_CHANNELS);
        fcmBuilder.setChannelMapping(0, 0, 0, 0);
        fcmBuilder.setChannelMapping(0, 1, 0, 1);
        fcmBuilder.setChannelMapping(0, 2, 1, 0);
        fcmBuilder.setChannelMapping(0, 3, 1, 1);

        fcmBuilder.setChannelMapping(1, 0, 2, 0);
        fcmBuilder.setChannelMapping(1, 1, 2, 1);
        fcmBuilder.setChannelMapping(1, 2, 3, 0);
        fcmBuilder.setChannelMapping(1, 3, 3, 1);
        fcm = fcmBuilder.build();
        rxPaddingLeft = std::vector<ChannelIdx>(fcm->getNumberOfLogicalFrames(), 0);
        rxPaddingRight = std::vector<ChannelIdx>(fcm->getNumberOfLogicalFrames(), 0);
    }

    constexpr static uint16_t N_FRAMES = 2;
    constexpr static uint16_t N_CHANNELS = 4;

    FrameChannelMapping::Handle fcm;
    std::vector<ChannelIdx> rxPaddingLeft;
    std::vector<ChannelIdx> rxPaddingRight;
};

TEST_F(ProbeImplFcmRemapTest, OneToOne) {
    // Given
    std::vector<std::vector<ChannelIdx>> adapterActiveChannels;

    for(uint16_t frame = 0; frame < N_FRAMES; ++frame) {
        std::vector<ChannelIdx> channels(N_CHANNELS, 0);
        std::iota(std::begin(channels), std::end(channels), 0);
        adapterActiveChannels.push_back(channels);
    }

    // Expect
    auto actualFcm = ProbeImpl::remapFcm(fcm, adapterActiveChannels, rxPaddingLeft, rxPaddingRight);

    auto actualNFrames = actualFcm->getNumberOfLogicalFrames();
    auto actualNChannels = actualFcm->getNumberOfLogicalChannels();

    EXPECT_EQ(actualNFrames, N_FRAMES);
    EXPECT_EQ(actualNChannels, N_CHANNELS);

    EXPECT_EQ(actualFcm->getLogical(0, 0), (std::pair<uint16, int8_t>(0, 0)));
    EXPECT_EQ(actualFcm->getLogical(0, 1), (std::pair<uint16, int8_t>(0, 1)));
    EXPECT_EQ(actualFcm->getLogical(0, 2), (std::pair<uint16, int8_t>(1, 0)));
    EXPECT_EQ(actualFcm->getLogical(0, 3), (std::pair<uint16, int8_t>(1, 1)));

    EXPECT_EQ(actualFcm->getLogical(1, 0), (std::pair<uint16, int8_t>(2, 0)));
    EXPECT_EQ(actualFcm->getLogical(1, 1), (std::pair<uint16, int8_t>(2, 1)));
    EXPECT_EQ(actualFcm->getLogical(1, 2), (std::pair<uint16, int8_t>(3, 0)));
    EXPECT_EQ(actualFcm->getLogical(1, 3), (std::pair<uint16, int8_t>(3, 1)));
}

TEST_F(ProbeImplFcmRemapTest, NonStandard) {
    // Given
    std::vector<std::vector<ChannelIdx>> adapterActiveChannels {
        {32, 33, 38, 35},
        {43, 41, 42, 40}
    };

    // Expect
    auto actualFcm = ProbeImpl::remapFcm(fcm, adapterActiveChannels, rxPaddingLeft, rxPaddingRight);

    auto actualNFrames = actualFcm->getNumberOfLogicalFrames();
    auto actualNChannels = actualFcm->getNumberOfLogicalChannels();

    EXPECT_EQ(actualNFrames, N_FRAMES);
    EXPECT_EQ(actualNChannels, N_CHANNELS);

    EXPECT_EQ(actualFcm->getLogical(0, 0), (std::pair<uint16, int8_t>(0, 0)));
    EXPECT_EQ(actualFcm->getLogical(0, 1), (std::pair<uint16, int8_t>(0, 1)));
    EXPECT_EQ(actualFcm->getLogical(0, 3), (std::pair<uint16, int8_t>(1, 0)));
    EXPECT_EQ(actualFcm->getLogical(0, 2), (std::pair<uint16, int8_t>(1, 1)));

    // Change
    EXPECT_EQ(actualFcm->getLogical(1, 3), (std::pair<uint16, int8_t>(2, 0)));
    EXPECT_EQ(actualFcm->getLogical(1, 1), (std::pair<uint16, int8_t>(2, 1)));
    EXPECT_EQ(actualFcm->getLogical(1, 2), (std::pair<uint16, int8_t>(3, 0)));
    EXPECT_EQ(actualFcm->getLogical(1, 0), (std::pair<uint16, int8_t>(3, 1)));
}

}


