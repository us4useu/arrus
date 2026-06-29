#include <gtest/gtest.h>
#include <vector>

#include "arrus/core/common/logging.h"
#include "TxTimeoutRegister.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"
#include "arrus/core/common/tests.h"

namespace {

using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;
using namespace arrus::framework;

std::vector<std::vector<float>> getDefaultRxDelays(size_t nSeqs, size_t nOps) {
    std::vector<std::vector<float>> result(nSeqs);
    for(auto &s: result){
        s.resize(nOps);
    }
    return result;
}


TEST(TxTimeoutRegisterFactoryTest, HandlesProperlyNoTxTimeouts) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);
    ops::us4r::Pulse pulse0{10.0e6f, 5000.0f, false}; // 500 us pulse

    //size_t nTimeouts = 0;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                // TOTAL TX TIME: 500 us + 10 us
                )
                ).getTxRx(),
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{0, [](float frequency) {return frequency;}, {}};
    auto reg = factory.createFor({sequence});
    EXPECT_TRUE(reg.empty());
}

/**
 * Sequence with TX ops and TX nops.
 */
TEST(TxTimeoutRegisterFactoryTest, HandlesProperlyOnlyTxNop) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    ops::us4r::Pulse pulse0{10.0e6f, 5000.0f, false}; // 500 us pulse

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                // TOTAL TX TIME: 500 us + 10 us
                )
                ).getTxRx(),
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{0, [](float frequency) {return frequency;}, getDefaultRxDelays(1, 1)};
    auto reg = factory.createFor({sequence});
    EXPECT_TRUE(reg.empty());
}

TEST(TxTimeoutRegisterFactoryTest, HandlesProperlyTxNopsWithTxOps) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    BitMask rxAperture(nChannels, true);
    ops::us4r::Pulse pulse0{1.0e6f, 300.0f, false}; // 500 us pulse

    std::vector<std::vector<float>> rxDelays = {{10e-6f, 20e-6f}};
    size_t nTimeouts = 3;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = BitMask(nChannels, true),
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                )
            ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = BitMask(nChannels, false),
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                )
            ).getTxRx(),
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{nTimeouts, [](float frequency) {return frequency;}, rxDelays};
    auto reg = factory.createFor({sequence});
    EXPECT_EQ(2, reg.getTimeouts().size());
    EXPECT_EQ(300 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(1));
    EXPECT_EQ(37 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(0));
    EXPECT_EQ(0, reg.getTimeoutId({SequenceId{0}, OpId{1}}));
    EXPECT_EQ(1, reg.getTimeoutId({SequenceId{0}, OpId{0}}));
}

TEST(TxTimeoutRegisterFactoryTest, CalculatesTxTimeoutsProperlySingleOp) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);
    ops::us4r::Pulse pulse0{1.0e6f, 300.0f, false}; // 300 us pulse
    size_t nTimeouts = 4;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                // TOTAL TX TIME: 300 us + 10 us
                )
                ).getTxRx(),
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{nTimeouts, [](float frequency) {return frequency;}, getDefaultRxDelays(1, 1)};
    auto reg = factory.createFor({sequence});
    EXPECT_EQ(1, reg.getTimeouts().size());
    EXPECT_EQ(300 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(0));
    EXPECT_EQ(0, reg.getTimeoutId({SequenceId{0}, OpId{0}}));
}

TEST(TxTimeoutRegisterFactoryTest, CalculatesTxTimeoutsProperlySingleOpLessThan1Us) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);
    ops::us4r::Pulse pulse0{10.0e6f, 2.0f, false}; // 500 us pulse
    size_t nTimeouts = 4;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                // TOTAL TX TIME: 0.2 us
                )
                ).getTxRx(),
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{nTimeouts, [](float frequency) {return frequency;}, getDefaultRxDelays(1, 1)};
    auto reg = factory.createFor({sequence});
    EXPECT_EQ(1, reg.getTimeouts().size());
    // Should be exactly EPSILON.
    EXPECT_EQ(TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(0));
    EXPECT_EQ(0, reg.getTimeoutId({SequenceId{0}, OpId{0}}));
}

TEST(TxTimeoutRegisterFactoryTest, CalculatesTxTimeoutsProperly3Ops) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    std::vector<float> delays1(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);

    ops::us4r::Pulse pulse0{1e6f, 500.0f, false}; // 500 us pulse
    ops::us4r::Pulse pulse1{1e6f, 1.0f, true}; // 1 us pulse

    float delay0 = 10e-6f;
    float delay1 = 1e-6f;
    delays0.at(21) = delay0;
    delays1.at(7) = delay1;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                // TOTAL TX TIME: 500 us + 10 us
            )
        ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays1,
                x.pulse = pulse1
                // TOTAL TX TIME: 1 us + 1 us
            )
        ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse1
                // TOTAL TX TIME: 10us + 1 us
            )
        ).getTxRx()
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{4, [](float frequency) {return frequency;}, getDefaultRxDelays(1, 3)};
    auto reg = factory.createFor({sequence});
    EXPECT_EQ(3, reg.getTimeouts().size());
    EXPECT_EQ(510 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(2));
    // upper limit: 11 us
    // 510 / 2 -> 255 / 2 -> 127 / 2 -> 63 / 2 -> 31 // 2 -> 15
    // + EPSILON
    EXPECT_EQ(15 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(1));
    // upper limit: 2 us
    // 15 / 2 -> 7 / 2 -> 3
    // + EPSILON
    EXPECT_EQ(3 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(0));
    EXPECT_EQ(2, reg.getTimeoutId({SequenceId{0}, OpId{0}}));
    EXPECT_EQ(0, reg.getTimeoutId({SequenceId{0}, OpId{1}}));
    EXPECT_EQ(1, reg.getTimeoutId({SequenceId{0}, OpId{2}}));
}

TEST(TxTimeoutRegisterFactoryTest, CalculatesTxTimeoutsProperly5Ops) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    std::vector<float> delays1(nChannels, 0.0f);
    std::vector<float> delays2(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);

    ops::us4r::Pulse pulse0{1.0e6f, 500.0f, false}; // 500 us pulse
    ops::us4r::Pulse pulse1{1.0e6f, 300.0f, false}; // 300 us pulse
    ops::us4r::Pulse pulse2{1.0e6f, 200.0f, true}; // 200 us
    ops::us4r::Pulse pulse3{9e6f, 200.0f, true}; //  ~20 us
    ops::us4r::Pulse pulse4{3.0e6f, 1.0f, true}; // 0.3 us

    float delay0 = 10e-6f;
    float delay1 = 1e-6f;
    delays0.at(21) = delay0;
    delays1.at(7) = delay1;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
                // TOTAL TX TIME: 500 us + 10 us
                )
                ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays1,
                x.pulse = pulse1
                // TOTAL TX TIME: 300 us + 1 us
                )
                ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse2
                // TOTAL TX TIME: 200 us + 10 us
            )
        ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays1,
                x.pulse = pulse3
                // TOTAL TX TIME: 20 us + 1 us
            )
        ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays2,
                x.pulse = pulse4
                // TOTAL TX TIME: 0.5 us + 0 us
            )
        ).getTxRx()
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{4, [](float frequency) {return frequency;}, getDefaultRxDelays(1, 5)};
    auto reg = factory.createFor({sequence});
    EXPECT_EQ(4, reg.getTimeouts().size());
    EXPECT_EQ(510 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(3));
    // upper limit: 301 us
    // So it will be the same as the previous one. Try with the next one.
    // upper limit: 210 us
    // 510 / 2 -> 255
    // + EPSILON
    EXPECT_EQ(255 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(2));
    // upper limit: 21 us
    // 255 / 2 -> 127 ... -> 31
    // + EPSILON
    EXPECT_EQ(31 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(1));
    // upper limit: 0.5 us
    // 31 / 2 -> .. 1
    // EPSILON
    EXPECT_EQ(1 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(0));

    EXPECT_EQ(3, reg.getTimeoutId({SequenceId{0}, OpId{0}}));
    EXPECT_EQ(3, reg.getTimeoutId({SequenceId{0}, OpId{1}}));
    EXPECT_EQ(2, reg.getTimeoutId({SequenceId{0}, OpId{2}}));
    EXPECT_EQ(1, reg.getTimeoutId({SequenceId{0}, OpId{3}}));
    EXPECT_EQ(0, reg.getTimeoutId({SequenceId{0}, OpId{4}}));
}


TEST(TxTimeoutRegisterFactoryTest, CalculatesTxTimeoutsProperly30psWithDynamicDelayProfiles) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    std::vector<float> delays1(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);

    ops::us4r::Pulse pulse{1e6f, 1.0f, true}; // 1 us pulse

    float delay0 = 10e-6f;
    float delay1 = 1e-6f;
    delays0.at(21) = delay0;
    delays1.at(7) = delay1;

    std::vector<TxRx> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse
                // TOTAL TX TIME: 100 us + 1 us
                )
                ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays1,
                x.pulse = pulse
                // TOTAL TX TIME: 300 us + 1 us
                )
                ).getTxRx(),
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse
                // TOTAL TX TIME: 10us + 1 us
                )
                ).getTxRx()
    };

    TxRxSequence sequence{txrxs, {}};

    // TX Delay profiles:

    // max delay is set in the following manner:
    // TX/RX 0: from the profile 0, max delay => 100 us
    // Tx/RX 1: from the profile 1, max delay => 300 us
    // TX/RX 2: from the TX/RX sequence, max delay 10 us

    // profile 0, TX/RX 0
    std::vector<float> delays00(nChannels, 0);
    // profile 0, TX/RX 1
    std::vector<float> delays01(nChannels, 0);
    std::vector<float> delays02(nChannels, 0);

    // profile 0, TX/RX 0, channel 22
    delays00.at(22) = 100e-6f;

    const auto delaysProfile0 = ::arrus::concat<float>({delays00, delays01, delays02});

    // profile 1
    std::vector<float> delays10(nChannels, 0);
    std::vector<float> delays11(nChannels, 0);
    std::vector<float> delays12(nChannels, 0);

    // profile 1, TX/RX 1, channel 1
    delays11.at(0) = 300e-6f;

    const auto delaysProfile1 = ::arrus::concat<float>({delays10, delays11, delays12});

    NdArray::Shape shape{txrxs.size(), nChannels};
    DeviceId placement{DeviceType::Us4R, 0};

    std::vector<NdArray> profiles = {
        NdArray::asarray<float>(delaysProfile0, shape, placement, "delays0"),
        NdArray::asarray<float>(delaysProfile1, shape, placement, "delays1")
    };

    std::unordered_map<std::string, ops::us4r::DelayProfiles> delayProfiles;

    delayProfiles.insert({sequence.getName(), profiles});

    TxTimeoutRegisterFactory factory{4, [](float frequency) {return frequency;}, getDefaultRxDelays(1, 3)};
    auto reg = factory.createFor({sequence}, delayProfiles);

    ASSERT_EQ(3, reg.getTimeouts().size());
    EXPECT_EQ(301 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(2));
    // upper limit: 11 us
    // 301 / 2 -> 150
    // + EPSILON
    EXPECT_EQ(150 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(1));
    // upper limit: 2 us
    // 150 / 2 -> 75 / 2 -> 37 / 2 -> 18
    // + EPSILON
    EXPECT_EQ(18 + TxTimeoutRegisterFactory::EPSILON, reg.getTimeouts().at(0));
    EXPECT_EQ(1, reg.getTimeoutId({SequenceId{0}, OpId{0}}));
    EXPECT_EQ(2, reg.getTimeoutId({SequenceId{0}, OpId{1}}));
    EXPECT_EQ(0, reg.getTimeoutId({SequenceId{0}, OpId{2}}));
}

}


int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}