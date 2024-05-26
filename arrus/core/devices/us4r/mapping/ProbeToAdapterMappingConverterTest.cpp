#include <ostream>
#include <gtest/gtest.h>

#include "ProbeToAdapterMappingConverter.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/probe/ProbeImpl.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"

namespace {

using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using ::arrus::devices::FrameChannelMappingAddress;

class ProbeImplFcmRemapTest : public ::testing::Test {
protected:
    void SetUp() override {

        FrameChannelMappingBuilder fcmBuilder(N_FRAMES, N_CHANNELS);
        fcmBuilder.setChannelMapping(0, 0, 0, 0, 0);
        fcmBuilder.setChannelMapping(0, 1, 0, 0, 1);
        fcmBuilder.setChannelMapping(0, 2, 0, 1, 0);
        fcmBuilder.setChannelMapping(0, 3, 0, 1, 1);

        fcmBuilder.setChannelMapping(1, 0, 0, 2, 0);
        fcmBuilder.setChannelMapping(1, 1, 0, 2, 1);
        fcmBuilder.setChannelMapping(1, 2, 0, 3, 0);
        fcmBuilder.setChannelMapping(1, 3, 0, 3, 1);
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
    ProbeToAdapterMappingConverter converter;
    auto actualFcm = ProbeImpl::remapFcm(fcm, adapterActiveChannels, rxPaddingLeft, rxPaddingRight);

    auto actualNFrames = actualFcm->getNumberOfLogicalFrames();
    auto actualNChannels = actualFcm->getNumberOfLogicalChannels();

    EXPECT_EQ(actualNFrames, N_FRAMES);
    EXPECT_EQ(actualNChannels, N_CHANNELS);

    EXPECT_EQ(actualFcm->getLogical(0, 0), (FrameChannelMappingAddress(0, 0, 0)));
    EXPECT_EQ(actualFcm->getLogical(0, 1), (FrameChannelMappingAddress(0, 0, 1)));
    EXPECT_EQ(actualFcm->getLogical(0, 2), (FrameChannelMappingAddress(0, 1, 0)));
    EXPECT_EQ(actualFcm->getLogical(0, 3), (FrameChannelMappingAddress(0, 1, 1)));

    EXPECT_EQ(actualFcm->getLogical(1, 0), (FrameChannelMappingAddress(0, 2, 0)));
    EXPECT_EQ(actualFcm->getLogical(1, 1), (FrameChannelMappingAddress(0, 2, 1)));
    EXPECT_EQ(actualFcm->getLogical(1, 2), (FrameChannelMappingAddress(0, 3, 0)));
    EXPECT_EQ(actualFcm->getLogical(1, 3), (FrameChannelMappingAddress(0, 3, 1)));
}
//
//TEST_F(ProbeImplFcmRemapTest, NonStandard) {
//    // Given
//    std::vector<std::vector<ChannelIdx>> adapterActiveChannels {
//        {32, 33, 38, 35},
//        {43, 41, 42, 40}
//    };
//
//    // Expect
//    auto actualFcm = ProbeImpl::remapFcm(fcm, adapterActiveChannels, rxPaddingLeft, rxPaddingRight);
//
//    auto actualNFrames = actualFcm->getNumberOfLogicalFrames();
//    auto actualNChannels = actualFcm->getNumberOfLogicalChannels();
//
//    EXPECT_EQ(actualNFrames, N_FRAMES);
//    EXPECT_EQ(actualNChannels, N_CHANNELS);
//
//    EXPECT_EQ(actualFcm->getLogical(0, 0), (FrameChannelMappingAddress(0, 0, 0)));
//    EXPECT_EQ(actualFcm->getLogical(0, 1), (FrameChannelMappingAddress(0, 0, 1)));
//    EXPECT_EQ(actualFcm->getLogical(0, 3), (FrameChannelMappingAddress(0, 1, 0)));
//    EXPECT_EQ(actualFcm->getLogical(0, 2), (FrameChannelMappingAddress(0, 1, 1)));
//
//    // Change
//    EXPECT_EQ(actualFcm->getLogical(1, 3), (FrameChannelMappingAddress(0, 2, 0)));
//    EXPECT_EQ(actualFcm->getLogical(1, 1), (FrameChannelMappingAddress(0, 2, 1)));
//    EXPECT_EQ(actualFcm->getLogical(1, 2), (FrameChannelMappingAddress(0, 3, 0)));
//    EXPECT_EQ(actualFcm->getLogical(1, 0), (FrameChannelMappingAddress(0, 3, 1)));
//}

}

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


// TODO implement the below test cases
//TEST_P(ProbeToadapterMappingConverterTest, OneToOne) {
//    converter = ProbeToAdapterMappingConverter(const DeviceId &probeTxId, const DeviceId &probeRxId, ProbeSettings probeTx,
//                    ProbeSettings probeRx, std::vector<ChannelIdx> txProbeMask,
//                    std::vector<ChannelIdx> rxProbeMask, const ChannelIdx adapterNChannels)
//
//}
//
//TEST_F(ProbeToadapterMappingConverterTest, SingleAdapterSubaperture) {
//
//}
//
//TEST_F(ProbeToadapterMappingConverterTest, MultipleAdapterSubapertures) {
//
//}
//
//TEST_F(ProbeToadapterMappingConverterTest, NonStandard) {
//
//}
