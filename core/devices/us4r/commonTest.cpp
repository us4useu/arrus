#include <gtest/gtest.h>
#include "common.h"

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


TxRxParameters getStdTxRxParameters(const std::vector<bool> &rxAperture) {
    return TxRxParameters(TX_APERTURE, TX_DELAYS, PULSE, rxAperture,
                          RX_SAMPLE_RANGE, DECIMATION_FACTOR, PRI);
}


void verify(const std::vector<TxRxParamsSequence> &expected,
            const std::vector<TxRxParamsSequence> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_EQ(expected, actual);
}

TEST(SplitRxApertureIfNecessaryTest, SplitsSingleOperationCorrectly) {
    std::vector<bool> rxAperture(128);
    rxAperture[1] = rxAperture[16] = rxAperture[33] = rxAperture[97] = true;

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    std::vector<TxRxParamsSequence> res = splitRxAperturesIfNecessary(in);

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
    verify(expected, res);
}

TEST(SplitRxApertureIfNecessaryTest, DoesNotSplitOpIfNotNecessary) {
    std::vector<bool> rxAperture(128);
    rxAperture[1] = rxAperture[16] = rxAperture[31] = true;

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    std::vector<TxRxParamsSequence> res = splitRxAperturesIfNecessary(in);

    std::vector<TxRxParamsSequence> expected{
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    verify(expected, res);
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
    std::vector<TxRxParamsSequence> res = splitRxAperturesIfNecessary(in);

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
    verify(expected, res);
}

TEST(SplitRxApertureIfNecessaryTest, SplitsFullRxApertureCorrectly) {
    std::vector<bool> rxAperture = getNTimes(true, 128);

    std::vector<TxRxParamsSequence> in = {
        {
            getStdTxRxParameters(rxAperture)
        }
    };
    std::vector<TxRxParamsSequence> res = splitRxAperturesIfNecessary(in);

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
    verify(expected, res);
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
    std::vector<TxRxParamsSequence> res = splitRxAperturesIfNecessary(in);

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
        expectedSeq0.push_back(TxRxParameters::US4OEM_NOP);
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
        expectedSeq1.push_back(TxRxParameters::US4OEM_NOP);
        expectedSeq1.push_back(getStdTxRxParameters(rxAperture2));
        expectedSeq1.push_back(getStdTxRxParameters(rxAperture3));
        expectedSeq1.push_back(getStdTxRxParameters(rxAperture4));
    }

    std::vector<TxRxParamsSequence> expected{expectedSeq0, expectedSeq1};
    verify(expected, res);
}

}