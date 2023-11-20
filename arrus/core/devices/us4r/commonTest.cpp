#include <gtest/gtest.h>
#include "common.h"

#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;

const Pulse PULSE(7e6, 3.5, true);
const Interval<uint32> RX_SAMPLE_RANGE(0, 4095);
const uint32 DECIMATION_FACTOR = 3;
const float PRI = 300e-6;
const std::vector<bool> TX_APERTURE = getNTimes(true, 128);
const std::vector<float> TX_DELAYS(128);
const std::vector<std::vector<uint8_t>> DEFAULT_MAPPING1({getRange<uint8_t>(0, 128)});
const std::vector<std::vector<uint8_t>> DEFAULT_MAPPING2({getRange<uint8_t>(0, 128), getRange<uint8_t>(0, 128)});

constexpr int32 FCM_UNAVAILABLE_VALUE = static_cast<int32>(FrameChannelMapping::UNAVAILABLE);

TxRxParameters getStdTxRxParameters(const std::vector<bool> &rxAperture) {
    return TxRxParameters(TX_APERTURE, TX_DELAYS, PULSE, rxAperture,
                          RX_SAMPLE_RANGE, DECIMATION_FACTOR, PRI);
}


void verifyOps(const std::vector<TxRxParamsSequence> &expected,
               const std::vector<TxRxParamsSequence> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_EQ(expected, actual);
}

#define ARRUS_SET_FCM(module, frame, channel, dstFrame, dstChannel) \
    expectedDstFrame(module, frame, channel) = dstFrame;       \
    expectedDstChannel(module, frame, channel) = dstChannel;        \


TEST(SplitRxApertureIfNecessaryTest, SplitsSingleOperationCorrectly) {
    std::vector<bool> rxAperture(128);
    rxAperture[1] = rxAperture[16] = rxAperture[33] = rxAperture[97] = true;

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> inputTxDelayProfiles;

    auto [res, fcmDstFrame, fcmDstChannel, outConstants] = splitRxAperturesIfNecessary(in, DEFAULT_MAPPING1, inputTxDelayProfiles);

    std::vector<bool> expectedRxAperture0(128);
    expectedRxAperture0[1] = true;
    expectedRxAperture0[16] = true;
    std::vector<bool> expectedRxAperture1(128);
    expectedRxAperture1[33] = true;
    std::vector<bool> expectedRxAperture2(128);
    expectedRxAperture2[97] = true;
    std::vector<TxRxParamsSequence> expected{
        {
            getStdTxRxParameters(expectedRxAperture0),
            getStdTxRxParameters(expectedRxAperture1),
            getStdTxRxParameters(expectedRxAperture2)
        }
    };
    verifyOps(expected, res);

    // FCM
    Eigen::Tensor<int32, 3> expectedDstFrame(1, 1, 4);
    Eigen::Tensor<int32, 3> expectedDstChannel(1, 1, 4);
    expectedDstFrame(0, 0, 0) = 0;
    expectedDstChannel(0, 0, 0) = 0;

    expectedDstFrame(0, 0, 1) = 0;
    expectedDstChannel(0, 0, 1) = 1;

    expectedDstFrame(0, 0, 2) = 1;
    expectedDstChannel(0, 0, 2) = 0;
    expectedDstFrame(0, 0, 3) = 2;
    expectedDstChannel(0, 0, 3) = 0;

    ARRUS_EXPECT_TENSORS_EQ(fcmDstFrame, expectedDstFrame);
    ARRUS_EXPECT_TENSORS_EQ(fcmDstChannel, expectedDstChannel);
}

TEST(SplitRxApertureIfNecessaryTest, DoesNotSplitOpIfNotNecessary) {
    std::vector<bool> rxAperture(128);
    rxAperture[1] = rxAperture[16] = rxAperture[31] = true;

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> inputTxDelayProfiles;

    auto [res, fcmDstFrame, fcmDstChannel, outConstants] = splitRxAperturesIfNecessary(in, DEFAULT_MAPPING1, inputTxDelayProfiles);

    std::vector<TxRxParamsSequence> expected{
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    verifyOps(expected, res);

    // FCM
    Eigen::Tensor<int32, 3> expectedDstFrame(1, 1, 3);
    Eigen::Tensor<int32, 3> expectedDstChannel(1, 1, 3);

    expectedDstFrame(0, 0, 0) = 0;
    expectedDstChannel(0, 0, 0) = 0;

    expectedDstFrame(0, 0, 1) = 0;
    expectedDstChannel(0, 0, 1) = 1;

    expectedDstFrame(0, 0, 2) = 0;
    expectedDstChannel(0, 0, 2) = 2;
    ARRUS_EXPECT_TENSORS_EQ(fcmDstFrame, expectedDstFrame);
    ARRUS_EXPECT_TENSORS_EQ(fcmDstChannel, expectedDstChannel);
}

TEST(SplitRxApertureIfNecessaryTest, SplitsMultipleOpsCorrectly) {
    std::vector<bool> rxAperture0(128);
    rxAperture0[1] = rxAperture0[16] = rxAperture0[33] = rxAperture0[80] = true;
    // expected output two apertures: (1, 16), (33, 80)
    std::vector<bool> rxAperture1(128);
    rxAperture1[0] = rxAperture1[96] = rxAperture1[16] = rxAperture1[48] = true;
    // expected two apertures: (0, 16), (48, 96)
    std::vector<bool> rxAperture2(128);
    rxAperture2[63] = true;
    // expected one aperture
    std::vector<bool> rxAperture3(128);
    rxAperture3[0] = rxAperture3[32] = rxAperture3[96] = rxAperture3[48] = true;
    // expected three apertures: (0, 48), (32), (96)

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture0),
            getStdTxRxParameters(rxAperture1),
            getStdTxRxParameters(rxAperture2),
            getStdTxRxParameters(rxAperture3)
        }
    };
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> inputTxDelayProfiles;
    auto [res, fcmDstFrame, fcmDstChannel, outChannels] = splitRxAperturesIfNecessary(in, DEFAULT_MAPPING1, inputTxDelayProfiles);

    // IN op 0
    std::vector<bool> expRxAperture0(128);
    expRxAperture0[1] = expRxAperture0[16] = true;
    std::vector<bool> expRxAperture1(128);
    expRxAperture1[33] = expRxAperture1[80] = true;
    // IN op 1
    std::vector<bool> expRxAperture2(128);
    expRxAperture2[0] = expRxAperture2[16] = true;
    std::vector<bool> expRxAperture3(128);
    expRxAperture3[48] = expRxAperture3[96] = true;
    // IN op 2
    std::vector<bool> expRxAperture4(128);
    expRxAperture4[63] = true;
    // In op 3
    std::vector<bool> expRxAperture5(128);
    expRxAperture5[0] = expRxAperture5[48] = true;
    std::vector<bool> expRxAperture6(128);
    expRxAperture6[32] = true;
    std::vector<bool> expRxAperture7(128);
    expRxAperture7[96] = true;

    std::vector<TxRxParamsSequence> expected{
        {
            getStdTxRxParameters(expRxAperture0),
            getStdTxRxParameters(expRxAperture1),
            getStdTxRxParameters(expRxAperture2),
            getStdTxRxParameters(expRxAperture3),
            getStdTxRxParameters(expRxAperture4),
            getStdTxRxParameters(expRxAperture5),
            getStdTxRxParameters(expRxAperture6),
            getStdTxRxParameters(expRxAperture7)
        }
    };
    verifyOps(expected, res);

    // FCM
    Eigen::Tensor<int32, 3> expectedDstFrame(1, 4, 4);
    Eigen::Tensor<int32, 3> expectedDstChannel(1, 4, 4);

    // frame 1.1:
    ARRUS_SET_FCM(0, 0, 0, 0, 0);
    ARRUS_SET_FCM(0, 0, 1, 0, 1);

    // frame 1.2:
    ARRUS_SET_FCM(0, 0, 2, 1, 0);
    ARRUS_SET_FCM(0, 0, 3, 1, 1);

    // frame 2.1:
    ARRUS_SET_FCM(0, 1, 0, 2, 0);
    ARRUS_SET_FCM(0, 1, 1, 2, 1);
    // frame 2.2:
    ARRUS_SET_FCM(0, 1, 2, 3, 0);
    ARRUS_SET_FCM(0, 1, 3, 3, 1);

    // frame 3:
    ARRUS_SET_FCM(0, 2, 0, 4, 0);
    // There is no rx channels > 0 for 3rd op.
    ARRUS_SET_FCM(0, 2, 1, 0, FCM_UNAVAILABLE_VALUE);
    ARRUS_SET_FCM(0, 2, 2, 0, FCM_UNAVAILABLE_VALUE);
    ARRUS_SET_FCM(0, 2, 3, 0, FCM_UNAVAILABLE_VALUE);

    // frame 4.1:
    // 0
    ARRUS_SET_FCM(0, 3, 0, 5, 0);
    // 32
    ARRUS_SET_FCM(0, 3, 1, 6, 0);
    // frame 4.2:
    // 48
    // NOTE! 48 is assigned to frame 5, because 32 mod 32 is already covered
    // by 0
    ARRUS_SET_FCM(0, 3, 2, 5, 1);
    // frame 4.3
    // 96
    ARRUS_SET_FCM(0, 3, 3, 7, 0);

    ARRUS_EXPECT_TENSORS_EQ(fcmDstFrame, expectedDstFrame);
    ARRUS_EXPECT_TENSORS_EQ(fcmDstChannel, expectedDstChannel);
}

TEST(SplitRxApertureIfNecessaryTest, SplitsFullRxApertureCorrectly) {
    std::vector<bool> rxAperture = getNTimes(true, 128);

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> inputTxDelayProfiles;
    auto [res, fcmDstFrame, fcmDstChannel, outConstants] = splitRxAperturesIfNecessary(in, DEFAULT_MAPPING1, inputTxDelayProfiles);

    std::vector<bool> expectedRxAperture0(128);
    for(size_t i = 0; i < 32; ++i) {
        expectedRxAperture0[i] = true;
    }
    std::vector<bool> expectedRxAperture1(128);
    for(size_t i = 32; i < 64; ++i) {
        expectedRxAperture1[i] = true;
    }
    std::vector<bool> expectedRxAperture2(128);
    for(size_t i = 64; i < 96; ++i) {
        expectedRxAperture2[i] = true;
    }
    std::vector<bool> expectedRxAperture3(128);
    for(size_t i = 96; i < 128; ++i) {
        expectedRxAperture3[i] = true;
    }
    std::vector<TxRxParamsSequence> expected{
        {
            getStdTxRxParameters(expectedRxAperture0),
            getStdTxRxParameters(expectedRxAperture1),
            getStdTxRxParameters(expectedRxAperture2),
            getStdTxRxParameters(expectedRxAperture3)
        }
    };
    verifyOps(expected, res);

    // FCM
    Eigen::Tensor<int32, 3> expectedDstFrame(1, 1, 128);
    Eigen::Tensor<int32, 3> expectedDstChannel(1, 1, 128);

    for(int32 i = 0; i < 32; ++i) {
        ARRUS_SET_FCM(0, 0, i, 0, i);
    }
    for(int32 i = 32; i < 64; ++i) {
        ARRUS_SET_FCM(0, 0, i, 1, i%32);
    }
    for(int32 i = 64; i < 96; ++i) {
        ARRUS_SET_FCM(0, 0, i, 2, i%32);
    }
    for(int32 i = 96; i < 128; ++i) {
        ARRUS_SET_FCM(0, 0, i, 3, i%32);
    }
    ARRUS_EXPECT_TENSORS_EQ(fcmDstFrame, expectedDstFrame);
    ARRUS_EXPECT_TENSORS_EQ(fcmDstChannel, expectedDstChannel);
}

// multiple sequences, each sequence should has the same size (padded with NOPs if necessary)
TEST(SplitRxApertureIfNecessaryTest, PadsWithNopsCorrectly) {
    // Two ops, first and the second should be padded
    TxRxParamsSequence seq0;
    {
        std::vector<bool> rxAperture0(128);
        rxAperture0[0] = rxAperture0[1] = rxAperture0[32] = true;
        std::vector<bool> rxAperture1(128);
        rxAperture1[16] = rxAperture1[17] = true;
        std::vector<bool> rxAperture2(128);
        rxAperture2[0] = rxAperture2[1] = true;

        seq0.push_back(getStdTxRxParameters(rxAperture0));
        seq0.push_back(getStdTxRxParameters(rxAperture1));
        seq0.push_back(getStdTxRxParameters(rxAperture2));
    }
    TxRxParamsSequence seq1;
    {
        std::vector<bool> rxAperture0(128);
        rxAperture0[0] = rxAperture0[1] = rxAperture0[34] = true;
        std::vector<bool> rxAperture1(128);
        rxAperture1[16] = rxAperture1[17] = true;
        std::vector<bool> rxAperture2(128);
        rxAperture2[16] = rxAperture2[48] = true;

        seq1.push_back(getStdTxRxParameters(rxAperture0));
        seq1.push_back(getStdTxRxParameters(rxAperture1));
        seq1.push_back(getStdTxRxParameters(rxAperture2));
    }

    std::vector<TxRxParamsSequence> in = {seq0, seq1};
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> inputTxDelayProfiles;
    auto [res, fcmDstFrame, fcmDstChannel, outConstants] = splitRxAperturesIfNecessary(in, DEFAULT_MAPPING2, inputTxDelayProfiles);

    TxRxParamsSequence expectedSeq0;
    {
        std::vector<bool> rxAperture0(128);
        rxAperture0[0] = rxAperture0[1] = true;

        std::vector<bool> rxAperture1(128);
        rxAperture1[32] = true;

        std::vector<bool> rxAperture2(128);
        rxAperture2[16] = rxAperture2[17] = true;

        std::vector<bool> rxAperture3(128);
        rxAperture3[0] = rxAperture3[1] = true;

        expectedSeq0.push_back(getStdTxRxParameters(rxAperture0));
        expectedSeq0.push_back(getStdTxRxParameters(rxAperture1));
        expectedSeq0.push_back(getStdTxRxParameters(rxAperture2));
        expectedSeq0.push_back(getStdTxRxParameters(rxAperture3));
        expectedSeq0.push_back(TxRxParameters::createRxNOPCopy(getStdTxRxParameters(rxAperture3)));
    }

    TxRxParamsSequence expectedSeq1;
    {
        std::vector<bool> rxAperture0(128);
        rxAperture0[0] = rxAperture0[1] = rxAperture0[34] = true;
        std::vector<bool> rxAperture2(128);
        rxAperture2[16] = rxAperture2[17] = true;
        std::vector<bool> rxAperture3(128);
        rxAperture3[16] = true;
        std::vector<bool> rxAperture4(128);
        rxAperture4[48] = true;

        expectedSeq1.push_back(getStdTxRxParameters(rxAperture0));
        expectedSeq1.push_back(TxRxParameters::createRxNOPCopy(getStdTxRxParameters(rxAperture0)));
        expectedSeq1.push_back(getStdTxRxParameters(rxAperture2));
        expectedSeq1.push_back(getStdTxRxParameters(rxAperture3));
        expectedSeq1.push_back(getStdTxRxParameters(rxAperture4));
    }

    std::vector<TxRxParamsSequence> expected{expectedSeq0, expectedSeq1};
    verifyOps(expected, res);

    // FCM
    Eigen::Tensor<int32, 3> expectedDstFrame(2, 3, 3);
    Eigen::Tensor<int32, 3> expectedDstChannel(2, 3, 3);
    expectedDstFrame.setZero();
    expectedDstChannel.setConstant(FCM_UNAVAILABLE_VALUE);
    // Module 0
    // Frame 1.1
    ARRUS_SET_FCM(0, 0, 0, 0, 0);
    ARRUS_SET_FCM(0, 0, 1, 0, 1);
    // Frame 1.2
    ARRUS_SET_FCM(0, 0, 2, 1, 0);
    // Frame 2
    ARRUS_SET_FCM(0, 1, 0, 2, 0);
    ARRUS_SET_FCM(0, 1, 1, 2, 1);
    // Frame 3
    ARRUS_SET_FCM(0, 2, 0, 3, 0);
    ARRUS_SET_FCM(0, 2, 1, 3, 1);

    // Module 1
    // Frame 1
    ARRUS_SET_FCM(1, 0, 0, 0, 0);
    ARRUS_SET_FCM(1, 0, 1, 0, 1);
    ARRUS_SET_FCM(1, 0, 2, 0, 2);
    // Frame 2
    ARRUS_SET_FCM(1, 1, 0, 1, 0);
    ARRUS_SET_FCM(1, 1, 1, 1, 1);
    // Frame 3.1
    ARRUS_SET_FCM(1, 2, 0, 2, 0);
    ARRUS_SET_FCM(1, 2, 1, 3, 0);

    ARRUS_EXPECT_TENSORS_EQ(fcmDstFrame, expectedDstFrame);
    ARRUS_EXPECT_TENSORS_EQ(fcmDstChannel, expectedDstChannel);
}

}