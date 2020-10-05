#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

#include "Us4OEMImpl.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/api/ops/us4r/tgc.h"

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;
using ::testing::_;


constexpr uint16 DEFAULT_PGA_GAIN = 30;
constexpr uint16 DEFAULT_LNA_GAIN = 24;

struct TestTxRxParams {

    TestTxRxParams() {
        for(int i = 0; i < 32; ++i) {
            rxAperture[i] = true;
        }
    }

    BitMask txAperture = getNTimes(true, Us4OEMImpl::N_TX_CHANNELS);;
    std::vector<float> txDelays = getNTimes(0.0f, Us4OEMImpl::N_TX_CHANNELS);
    ops::us4r::Pulse pulse{2.0e6f, 2.5f, true};
    BitMask rxAperture = getNTimes(false, Us4OEMImpl::N_ADDR_CHANNELS);
    uint32 decimationFactor = 1;
    float pri = 100e-6f;
    Interval<uint32> sampleRange{0, 4095};

    [[nodiscard]] TxRxParameters getTxRxParameters() const {
        return TxRxParameters(txAperture, txDelays, pulse,
                              rxAperture, sampleRange,
                              decimationFactor, pri);
    }
};


class Us4OEMImplEsaote3LikeTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        ius4oemPtr = dynamic_cast<MockIUs4OEM *>(ius4oem.get());
        BitMask activeChannelGroups = {true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true};
        std::vector<uint8> channelMapping = getRange<uint8>(0, 128);
        uint16 pgaGain = DEFAULT_PGA_GAIN;
        uint16 lnaGain = DEFAULT_LNA_GAIN;
        us4oem = std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            std::move(ius4oem), activeChannelGroups,
            channelMapping,
            pgaGain, lnaGain
        );
    }

    MockIUs4OEM *ius4oemPtr;
    Us4OEMImpl::Handle us4oem;
    TGCCurve defaultTGCCurve;
};


// ------------------------------------------ TESTING VALIDATION
TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidApertureSize) {
    // Tx aperture.
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txAperture = getNTimes(true, Us4OEMImpl::N_TX_CHANNELS + 1)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

    // Rx aperture: total size
    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = getNTimes(true, Us4OEMImpl::N_TX_CHANNELS - 1)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

//     Rx aperture: number of active elements
    BitMask rxAperture(128, false);
    for(size_t i = 0; i < 33; ++i) {
        rxAperture[i] = true;
    }
    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

    // Tx delays
    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(0.0f, Us4OEMImpl::N_TX_CHANNELS / 2)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidTxDelays) {
    // Tx delays
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(Us4OEMImpl::MAX_TX_DELAY + 1e-6f, Us4OEMImpl::N_TX_CHANNELS)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(Us4OEMImpl::MIN_TX_DELAY - 1e-6f, Us4OEMImpl::N_TX_CHANNELS)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidPri) {
    // Tx delays
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pri = Us4OEMImpl::MAX_PRI + 1e-6f))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pri = Us4OEMImpl::MIN_PRI - 1e-6f))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidNPeriodsOnly) {
    // Tx delays
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 1.3f, false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 1.5f, false)))
            .getTxRxParameters()
    };
    us4oem->setTxRxSequence(seq, defaultTGCCurve);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidFrequency) {
    // Tx delays
    const auto maxFreq = Us4OEMImpl::MAX_TX_FREQUENCY;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(std::nextafter(maxFreq, maxFreq + 1e6f), 1.0f, false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);

    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(Us4OEMImpl::MIN_TX_FREQUENCY - 0.5e6f, 1.0f, false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(us4oem->setTxRxSequence(seq, defaultTGCCurve),
                 IllegalArgumentException);
}
// TODO test memory overflow protection
// ------------------------------------------ Testing parameters set to IUs4OEM

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxMapping032) {
    // Rx aperture 0-32
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    std::vector<uint8> expectedRxMapping = getRange<uint8>(0, 32);
    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping, 0));

    us4oem->setTxRxSequence(seq, defaultTGCCurve);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxMapping032Missing1518) {
    // Rx aperture 0-32
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);
    rxAperture[15] = false;
    rxAperture[18] = false;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    std::vector<uint8> expectedRxMapping = getRange<uint8>(0, 32);
    // 0, 1, 2, .., 14, 16, 17, 19, 20, ..., 29, 15, 18
    setValuesInRange<uint8>(expectedRxMapping, 0, 15,
                            [](size_t i) { return (uint8) (i); });
    setValuesInRange<uint8>(expectedRxMapping, 15, 17,
                            [](size_t i) { return (uint8) (i + 1); });
    setValuesInRange<uint8>(expectedRxMapping, 17, 30,
                            [](size_t i) { return (uint8) (i + 2); });
    expectedRxMapping[30] = 15;
    expectedRxMapping[31] = 18;

    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping, 0));

    us4oem->setTxRxSequence(seq, defaultTGCCurve);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxMapping1648) {
    // Rx aperture 0-32
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 16, 48, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    std::vector<uint8> expectedRxMapping(32, 0);
    setValuesInRange<uint8>(expectedRxMapping, 0, 16,
                            [](size_t i) { return static_cast<uint8>(i + 16); });
    setValuesInRange<uint8>(expectedRxMapping, 16, 32,
                            [](size_t i) { return static_cast<uint8>(i % 16); });
    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping, 0));

    us4oem->setTxRxSequence(seq, defaultTGCCurve);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfMappings) {
    // Rx aperture 0-32
    BitMask rxAperture1(128, false);
    setValuesInRange(rxAperture1, 0, 32, true);
    BitMask rxAperture2(128, false);
    setValuesInRange(rxAperture2, 16, 48, true);
    BitMask rxAperture3(128, false);
    setValuesInRange(rxAperture3, 32, 64, true);

    std::vector<TxRxParameters> seq = {
        // 1st tx/rx
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture1))
            .getTxRxParameters(),
        // 2nd tx/rx
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture2))
            .getTxRxParameters(),
        // 3rd tx/rx
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture3))
            .getTxRxParameters()
    };
    std::vector<uint8> expectedRxMapping1 = getRange<uint8>(0, 32);
    std::vector<uint8> expectedRxMapping2(32, 0);
    setValuesInRange<uint8>(expectedRxMapping2, 0, 16,
                            [](size_t i) { return static_cast<uint8>(i + 16); });
    setValuesInRange<uint8>(expectedRxMapping2, 16, 32,
                            [](size_t i) { return static_cast<uint8>(i % 16); });

    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping1, 0));
    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping2, 1));

    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(0, _, _, _, _, 0, _));
    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(1, _, _, _, _, 1, _));
    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(2, _, _, _, _, 0, _));

    us4oem->setTxRxSequence(seq, defaultTGCCurve);
}

class Us4OEMImplConflictingChannelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        ius4oemPtr = dynamic_cast<MockIUs4OEM *>(ius4oem.get());
        BitMask activeChannelGroups = {true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true};
        // Esaote 2 Us4OEM:0 channel mapping
        std::vector<uint8> channelMapping = castTo<uint8, uint32>({26, 27, 25, 23, 28, 22, 20, 21,
                                                                   24, 18, 19, 15, 17, 16, 29, 13,
                                                                   11, 14, 30, 8, 12, 5, 10, 9,
                                                                   31, 7, 3, 6, 0, 2, 4, 1,
                                                                   56, 55, 54, 53, 57, 52, 51, 49,
                                                                   50, 48, 47, 46, 44, 45, 58, 42,
                                                                   43, 59, 40, 41, 60, 38, 61, 39,
                                                                   62, 34, 37, 63, 36, 35, 32, 33,
                                                                   92, 93, 89, 91, 88, 90, 87, 85,
                                                                   86, 84, 83, 82, 81, 80, 79, 77,
                                                                   78, 76, 95, 75, 74, 94, 73, 72,
                                                                   70, 64, 71, 68, 65, 69, 67, 66,
                                                                   96, 97, 98, 99, 100, 101, 102, 103,
                                                                   104, 105, 106, 107, 108, 109, 110, 111,
                                                                   112, 113, 114, 115, 116, 117, 118, 119,
                                                                   120, 121, 122, 123, 124, 125, 126, 127});
        uint16 pgaGain = DEFAULT_PGA_GAIN;
        uint16 lnaGain = DEFAULT_LNA_GAIN;
        us4oem = std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            std::move(ius4oem), activeChannelGroups,
            channelMapping,
            pgaGain, lnaGain
        );
    }

    MockIUs4OEM *ius4oemPtr;
    Us4OEMImpl::Handle us4oem;
    TGCCurve defaultTGCCurve;
};

TEST_F(Us4OEMImplConflictingChannelsTest, TurnsOffConflictingChannels) {
    BitMask rxAperture(128, false);

    // 10, 12, 14 and are conflicting:

    // (11, 14, 30,  8, 12,  5, 10,  9,
    //  31,  7,  3,  6,  0,  2,  4,  1,
    //  24, 23, 22, 21, 25, 20, 19, 17,
    //  18, 16, 15, 14, 12, 13, 26, 10)
    setValuesInRange(rxAperture, 16, 48, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };

    std::bitset<Us4OEMImpl::N_ADDR_CHANNELS> expectedRxAperture;
    setValuesInRange(expectedRxAperture, 16, 48, true);
    expectedRxAperture[43] = false;
    expectedRxAperture[44] = false;
    expectedRxAperture[47] = false;
    EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxAperture, 0));

    std::vector<uint8> expectedRxMapping = {11, 14, 30, 8, 12, 5, 10, 9,
                                            31, 7, 3, 6, 0, 2, 4, 1,
                                            24, 23, 22, 21, 25, 20, 19, 17,
                                            18, 16, 15, 13, 26,
        // not used:
                                            27, 28, 29};
    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping, 0));

    us4oem->setTxRxSequence(seq, defaultTGCCurve);
}

//TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxTimeAndDelay1) {
//    // Sample range -> rx delay
//    // end-start / sampling frequency
//    Interval<uint32> sampleRange(0, 1000);
//
//    std::vector<TxRxParameters> seq1 = {
//        ARRUS_STRUCT_INIT_LIST(
//            TestTxRxParams,
//            (x.sampleRange = sampleRange))
//            .getTxRxParameters()
//    };
//    EXPECT_CALL(*ius4oemPtr, SetRxDelay(Us4OEMImpl::RX_DELAY, 0));
//    // > number of
//    float minimumRxTime = (sampleRange.end() - sampleRange.start()) / Us4OEMImpl::SAMPLING_FREQUENCY;
//    EXPECT_CALL(*ius4oemPtr, SetRxTime());
//    // ScheduleReceive: starting sample
//    us4oem->setTxRxSequence(seq1, defaultTGCCurve);
//}

// sample range, rx time
// tx half periods
// TGC, interpolation to the destination values
// active channel groups! (NOP, no NOP)
// Test NOP handling
// Test generated FrameMapping

}

int main(int argc, char **argv) {
    std::cerr << "Starting" << std::endl;
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}