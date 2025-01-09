#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

#include "Us4OEMImpl.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;
using ::testing::_;
using ::testing::Ge;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

MATCHER_P(FloatNearPointwise, tol, "") {
    return std::abs(std::get<0>(arg) - std::get<1>(arg)) < tol;
}

constexpr uint16 DEFAULT_PGA_GAIN = 30;
constexpr uint16 DEFAULT_LNA_GAIN = 24;
constexpr float MAX_TX_FREQUENCY = 65e6f;
constexpr float MIN_TX_FREQUENCY = 1e6f;
constexpr uint32_t TX_OFFSET = 123;

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
    float pri = 200e-6f;
    Interval<uint32> sampleRange{0, 4096};

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
        // Default values returned by us4oem.
        ON_CALL(*ius4oemPtr, GetMaxTxFrequency).WillByDefault(testing::Return(MAX_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetMinTxFrequency).WillByDefault(testing::Return(MIN_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetTxOffset).WillByDefault(testing::Return(TX_OFFSET));

        BitMask activeChannelGroups = {true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true};
        std::vector<uint8> channelMapping = getRange<uint8>(0, 128);
        uint16 lnaGain = DEFAULT_LNA_GAIN;
        RxSettings rxSettings(std::nullopt, DEFAULT_PGA_GAIN, DEFAULT_LNA_GAIN, {}, 15'000'000, std::nullopt, true);
        us4oem = std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            std::move(ius4oem), activeChannelGroups,
            channelMapping, rxSettings,
            std::unordered_set<uint8>(),
            Us4OEMSettings::ReprogrammingMode::SEQUENTIAL,
            false,
            false
        );
    }

    MockIUs4OEM *ius4oemPtr;
    Us4OEMImpl::Handle us4oem;
    TGCCurve defaultTGCCurve;
    uint16 defaultRxBufferSize = 1;
    uint16 defaultBatchSize = 1;
    std::optional<float> defaultSri = std::nullopt;
    arrus::ops::us4r::Scheme::WorkMode defaultWorkMode = arrus::ops::us4r::Scheme::WorkMode::SYNC;
};


#define SET_TX_RX_SEQUENCE_TGC(us4oem, seq, tgc) \
     us4oem->setTxRxSequence(seq, tgc, defaultRxBufferSize, defaultBatchSize, defaultSri, defaultWorkMode)

#define SET_TX_RX_SEQUENCE(us4oem, seq) SET_TX_RX_SEQUENCE_TGC(us4oem, seq, defaultTGCCurve)

// ------------------------------------------ TESTING VALIDATION
TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidApertureSize) {
    // Tx aperture.
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txAperture = getNTimes(true, Us4OEMImpl::N_TX_CHANNELS + 1)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq), IllegalArgumentException);

    // Rx aperture: total size
    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = getNTimes(true, Us4OEMImpl::N_TX_CHANNELS - 1)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
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
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
                 IllegalArgumentException);

    // Tx delays
    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(0.0f, Us4OEMImpl::N_TX_CHANNELS / 2)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
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
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
                 IllegalArgumentException);

    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(Us4OEMImpl::MIN_TX_DELAY - 1e-6f, Us4OEMImpl::N_TX_CHANNELS)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
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
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
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
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
                 IllegalArgumentException);

    std::vector<TxRxParameters> seq2 = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 33.0f, false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq2), IllegalArgumentException);
    
    // The correct one.
    std::vector<TxRxParameters> seq3 = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 1.5f, false)))
            .getTxRxParameters()
    };
    SET_TX_RX_SEQUENCE(us4oem, seq3);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidNPeriodsCustomMaxLengthOEMPlus) {
    float maxPulseLength = 140e-6;
    ON_CALL(*ius4oemPtr, GetOemVersion).WillByDefault(testing::Return(2)); // OEM+

    us4oem->setMaximumPulseLength(maxPulseLength);

    float pulseLength = 100e-6;
    float frequency = 8e6;
    float nPeriods = std::roundf(pulseLength*frequency);
    // Correct
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(frequency, nPeriods, false)))
            .getTxRxParameters()
    };
    SET_TX_RX_SEQUENCE(us4oem, seq); // no throw expected
    
    // Incorrect
    std::vector<TxRxParameters> seq2 = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(frequency, std::roundf(frequency*150e-6), false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq2), IllegalArgumentException);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidNPeriodsCustomMaxLengthLegacyOEM) {
    float maxPulseLength = 140e-6;
    float frequency = 8e6;
    ON_CALL(*ius4oemPtr, GetOemVersion).WillByDefault(testing::Return(1)); // OEM

    EXPECT_THROW(us4oem->setMaximumPulseLength(maxPulseLength), IllegalArgumentException);
    // Incorrect
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(frequency, std::roundf(frequency*100e-6), false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq), IllegalArgumentException);
}


TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidFrequency) {
    // Tx delays
    const auto maxFreq = MAX_TX_FREQUENCY;
    const auto minFreq = MIN_TX_FREQUENCY;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(std::nextafter(maxFreq, maxFreq + 1e6f), 1.0f, false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq), IllegalArgumentException);

    seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(minFreq - 0.5e6f, 1.0f, false)))
            .getTxRxParameters()
    };
    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq), IllegalArgumentException);
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

    SET_TX_RX_SEQUENCE(us4oem, seq);
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

    SET_TX_RX_SEQUENCE(us4oem, seq);
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

    SET_TX_RX_SEQUENCE(us4oem, seq);
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

    SET_TX_RX_SEQUENCE(us4oem, seq);
}

class Us4OEMImplConflictingChannelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        ius4oemPtr = dynamic_cast<MockIUs4OEM *>(ius4oem.get());
        ON_CALL(*ius4oemPtr, GetMaxTxFrequency).WillByDefault(testing::Return(MAX_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetMinTxFrequency).WillByDefault(testing::Return(MIN_TX_FREQUENCY));
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

        RxSettings rxSettings(std::nullopt, DEFAULT_PGA_GAIN, DEFAULT_LNA_GAIN, {}, 15'000'000, std::nullopt, true);
        us4oem = std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            std::move(ius4oem), activeChannelGroups,
            channelMapping, rxSettings,
            std::unordered_set<uint8>(),
            Us4OEMSettings::ReprogrammingMode::SEQUENTIAL,
            false,
            false
        );
    }

    MockIUs4OEM *ius4oemPtr;
    Us4OEMImpl::Handle us4oem;
    TGCCurve defaultTGCCurve;
    uint16 defaultRxBufferSize = 1;
    uint16 defaultBatchSize = 1;
    std::optional<float> defaultSri = std::nullopt;
    arrus::ops::us4r::Scheme::WorkMode defaultWorkMode = arrus::ops::us4r::Scheme::WorkMode::SYNC;
};

TEST_F(Us4OEMImplConflictingChannelsTest, TurnsOffConflictingChannels) {
    BitMask rxAperture(128, false);

    //  11, 14, 30, 8, 12, 5, 10, 9,
    //  31, 7, 3, 6, 0, 2, 4, 1,
    //  56, 55, 54, 53, 57, 52, 51, 49,
    //  50, 48, 47, 46, 44, 45, 58, 42,

    // 10 (10, 42), 12 (12, 44), 14 (14, 46) are conflicting:

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

    // The channel mapping should stay unmodified
    // 27, 28, 29 are not used (should be turned off)
    std::vector<uint8> expectedRxMapping = {11, 14, 30, 8, 12, 5, 10, 9,
                                            31, 7, 3, 6, 0, 2, 4, 1,
                                            24, 23, 22, 21, 25, 20, 19, 17,
                                            18, 16, 15, 27, 28, 13, 26, 29};

    EXPECT_CALL(*ius4oemPtr, SetRxChannelMapping(expectedRxMapping, 0));

    SET_TX_RX_SEQUENCE(us4oem, seq);
}

// active channel groups! (NOP, no NOP)

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxTimeAndDelay1) {
    // Sample range -> rx delay
    // end-start / sampling frequency
    Interval<uint32> sampleRange(0, 1024);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.sampleRange = sampleRange))
            .getTxRxParameters()
    };
    ON_CALL(*ius4oemPtr, GetOemVersion).WillByDefault(testing::Return(2)); // OEM+
    EXPECT_CALL(*ius4oemPtr, SetRxDelay(0.0f, 0)); // Note: the value 0.0 is the default value
    uint32 nSamples = sampleRange.end() - sampleRange.start();
    float minimumRxTime = float(nSamples) / Us4OEMImpl::SAMPLING_FREQUENCY;
    EXPECT_CALL(*ius4oemPtr, SetRxTime(Ge(minimumRxTime), 0));
    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(0, _, nSamples, TX_OFFSET + sampleRange.start(), _, _, _));
    // ScheduleReceive: starting sample
    SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxTimeAndDelay2) {
    // Sample range -> rx delay
    // end-start / sampling frequency
    Interval<uint32> sampleRange(40, 1024 + 40);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.sampleRange = sampleRange))
            .getTxRxParameters()
    };
    ON_CALL(*ius4oemPtr, GetOemVersion).WillByDefault(testing::Return(2)); // OEM+
    EXPECT_CALL(*ius4oemPtr, SetRxDelay(0.0f, 0)); // Note: the default value of TxRxParameters
    uint32 nSamples = sampleRange.end() - sampleRange.start();
    float minimumRxTime = float(nSamples) / Us4OEMImpl::SAMPLING_FREQUENCY;
    EXPECT_CALL(*ius4oemPtr, SetRxTime(Ge(minimumRxTime), 0));
    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(0, _, nSamples, TX_OFFSET + sampleRange.start(), _, _, _));
    // ScheduleReceive: starting sample
    SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfTxHalfPeriods) {
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(3e6f, 1.5, true)))
            .getTxRxParameters()
    };
    EXPECT_CALL(*ius4oemPtr, SetTxHalfPeriods(3, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxFreqency(3e6f, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxInvert(true, 0));
    SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfTxHalfPeriods2) {
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(3e6, 3, false)))
            .getTxRxParameters()
    };
    EXPECT_CALL(*ius4oemPtr, SetTxHalfPeriods(6, 0));
    SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfTxHalfPeriods3) {
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(3e6, 30.5, false)))
            .getTxRxParameters()
    };
    EXPECT_CALL(*ius4oemPtr, SetTxHalfPeriods(61, 0));
    SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, TurnsOffTGCWhenEmpty) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().getTxRxParameters()
    };
    EXPECT_CALL(*ius4oemPtr, TGCDisable);
    SET_TX_RX_SEQUENCE_TGC(us4oem, seq, {});
}

TEST_F(Us4OEMImplEsaote3LikeTest, InterpolatesToTGCCharacteristicCorrectly) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().getTxRxParameters()
    };
    TGCCurve tgc = {14.000f, 14.001f, 14.002f};

    EXPECT_CALL(*ius4oemPtr, TGCEnable);

    TGCCurve expectedTgc = {14.0f, 15.0f, 16.0f};
    // normalized
    for(float &i : expectedTgc) {
        i = (i - 14.0f) / 40.f;
    }
    EXPECT_CALL(*ius4oemPtr, TGCSetSamples(Pointwise(FloatNearPointwise(1e-4), expectedTgc), _));

    SET_TX_RX_SEQUENCE_TGC(us4oem, seq, tgc);
}

TEST_F(Us4OEMImplEsaote3LikeTest, InterpolatesToTGCCharacteristicCorrectly2) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().getTxRxParameters()
    };
    TGCCurve tgc = {14.000f, 14.0005f, 14.001f};

    EXPECT_CALL(*ius4oemPtr, TGCEnable);

    TGCCurve expectedTgc = {14.0f, 14.5f, 15.0f};
    // normalized
    for(float &i : expectedTgc) {
        i = (i - 14.0f) / 40.f;
    }
    EXPECT_CALL(*ius4oemPtr, TGCSetSamples(Pointwise(FloatNearPointwise(1e-4), expectedTgc), _));
    SET_TX_RX_SEQUENCE_TGC(us4oem, seq, tgc);
}

TEST_F(Us4OEMImplEsaote3LikeTest, InterpolatesToTGCCharacteristicCorrectly3) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().getTxRxParameters()
    };
    TGCCurve tgc = {14.000f, 14.0002f, 14.0007f, 14.001f, 14.0015f};

    EXPECT_CALL(*ius4oemPtr, TGCEnable);

    TGCCurve expectedTgc = {14.0f, 14.2f, 14.7f, 15.0f, 15.5f};
    // normalized
    for(float &i : expectedTgc) {
        i = (i - 14.0f) / 40.f;
    }
    EXPECT_CALL(*ius4oemPtr, TGCSetSamples(Pointwise(FloatNearPointwise(1e-4), expectedTgc), _));
    SET_TX_RX_SEQUENCE_TGC(us4oem, seq, tgc);
}

TEST_F(Us4OEMImplEsaote3LikeTest, TurnsOffAllChannelsForNOP) {
    std::vector<TxRxParameters> seq = {
        TxRxParameters::US4OEM_NOP
    };
    // empty
    std::bitset<Us4OEMImpl::N_ADDR_CHANNELS> rxAperture, txAperture;
    // empty
    std::bitset<Us4OEMImpl::N_ACTIVE_CHANNEL_GROUPS> activeChannelGroup;
    EXPECT_CALL(*ius4oemPtr, SetRxAperture(rxAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxAperture(txAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetActiveChannelGroup(activeChannelGroup, 0));

    SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, TestFrameChannelMappingForNonconflictingRxMapping) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(us4oem, seq);

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);

    for(size_t i = 0; i < Us4OEMImpl::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), i);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

TEST_F(Us4OEMImplEsaote3LikeTest, TestFrameChannelMappingForNonconflictingRxMapping2) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 16, 48, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(us4oem, seq);

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);

    for(size_t i = 0; i < Us4OEMImpl::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), i);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

TEST_F(Us4OEMImplEsaote3LikeTest, TestFrameChannelMappingIncompleteRxAperture) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);

    rxAperture[31] = rxAperture[15] = false;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture))
            .getTxRxParameters()
    };
    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(us4oem, seq);

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);

    for(size_t i = 0; i < 30; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), i);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

TEST_F(Us4OEMImplConflictingChannelsTest, TestFrameChannelMappingForConflictingMapping) {
    BitMask rxAperture(128, false);
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
    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(us4oem, seq);

    for(size_t i = 0; i < Us4OEMImpl::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        std::cerr << (int16) address.getChannel() << ", ";
    }
    std::cerr << std::endl;

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);
    // turned off channels should be zeroed, so we just expect 0-31 here
    std::vector<int8> expectedDstChannels = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    };

    for(size_t i = 0; i < Us4OEMImpl::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), expectedDstChannels[i]);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

// ------------------------------------------ TESTING CHANNEL MASKING

class Us4OEMImplEsaote3ChannelsMaskTest : public ::testing::Test {
protected:
    void SetUp() override {
        ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        ius4oemPtr = dynamic_cast<MockIUs4OEM *>(ius4oem.get());
        ON_CALL(*ius4oemPtr, GetMaxTxFrequency).WillByDefault(testing::Return(MAX_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetMinTxFrequency).WillByDefault(testing::Return(MIN_TX_FREQUENCY));
    }

    Us4OEMImpl::Handle createHandle(const std::unordered_set<uint8> &channelsMask) {
        // This function can be called only once.

        BitMask activeChannelGroups = {true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true};

        std::vector<uint8> channelMapping = getRange<uint8>(0, 128);

        RxSettings rxSettings(std::nullopt, DEFAULT_PGA_GAIN, DEFAULT_LNA_GAIN, {}, 15'000'000, std::nullopt, true);
        return std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            // NOTE: due to the below move this function can be called only once
            std::move(ius4oem), activeChannelGroups,
            channelMapping, rxSettings, channelsMask,
            Us4OEMSettings::ReprogrammingMode::SEQUENTIAL,
            false,
            false
            );

    }

    std::unique_ptr<IUs4OEM> ius4oem;
    MockIUs4OEM *ius4oemPtr;
    TGCCurve defaultTGCCurve;
    uint16 defaultRxBufferSize = 1;
    uint16 defaultBatchSize = 1;
    std::optional<float> defaultSri = std::nullopt;
    arrus::ops::us4r::Scheme::WorkMode defaultWorkMode = arrus::ops::us4r::Scheme::WorkMode::SYNC;
};

// no masking - no channels are turned off
TEST_F(Us4OEMImplEsaote3ChannelsMaskTest, DoesNothingWithAperturesWhenNoChannelMask) {
    auto us4oem = createHandle(std::unordered_set<uint8>({}));

    BitMask rxAperture(128, false);
    BitMask txAperture(128, false);
    std::vector<float> txDelays(128, 0.0);

    txAperture[0] = txAperture[6] = txAperture[31] = txAperture[59] = true;
    txDelays[0] = 1e-6;
    txDelays[6] = 2e-6;
    txDelays[31] = 4e-6;
    txDelays[59] = 8e-6;
    rxAperture[0] = rxAperture[7] = rxAperture[31] = rxAperture[60] = true;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.rxAperture = rxAperture,
                x.txAperture = txAperture,
                x.txDelays = txDelays
            ))
            .getTxRxParameters()
    };

    auto expectedTxAperture = ::arrus::toBitset<Us4OEMImpl::N_ADDR_CHANNELS>(txAperture);
    auto expectedRxAperture = ::arrus::toBitset<Us4OEMImpl::N_ADDR_CHANNELS>(rxAperture);
    auto &expectedTxDelays = txDelays;

    EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxAperture(expectedTxAperture, 0));
    for(int i = 0; i < expectedTxDelays.size(); ++i) {
        EXPECT_CALL(*ius4oemPtr, SetTxDelay(i, expectedTxDelays[i], 0, 0));
    }

    SET_TX_RX_SEQUENCE(us4oem, seq);
}


TEST_F(Us4OEMImplEsaote3ChannelsMaskTest, MasksProperlyASingleChannel) {
    auto us4oem = createHandle(std::unordered_set<uint8>({7}));

    BitMask rxAperture(128, false);
    BitMask txAperture(128, false);
    std::vector<float> txDelays(128, 0.0);

    txAperture[0] = txAperture[7] = txAperture[33] = txAperture[95] = true;
    txDelays[0] = txDelays[7] = txDelays[33] = txDelays[95] = 1e-6;
    rxAperture[0] = rxAperture[7] = rxAperture[31] = rxAperture[60] = true;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.rxAperture = rxAperture,
                x.txAperture = txAperture,
                x.txDelays = txDelays
            ))
            .getTxRxParameters()
    };

    auto expectedTxAperture = ::arrus::toBitset<Us4OEMImpl::N_ADDR_CHANNELS>(txAperture);
    expectedTxAperture[7] = false;
    auto expectedRxAperture = ::arrus::toBitset<Us4OEMImpl::N_ADDR_CHANNELS>(rxAperture);
    expectedRxAperture[7] = false;

    std::vector<float> expectedTxDelays(txDelays);
    expectedTxDelays[7] = 0.0f;

    EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxAperture(expectedTxAperture, 0));
    for(int i = 0; i < expectedTxDelays.size(); ++i) {
        EXPECT_CALL(*ius4oemPtr, SetTxDelay(i, expectedTxDelays[i], 0, 0));
    }

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(us4oem, seq);

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);
    ASSERT_EQ(fcm->getNumberOfLogicalChannels(), Us4OEMImpl::N_RX_CHANNELS);

    std::vector<int8> expectedSrcChannels(Us4OEMImpl::N_RX_CHANNELS, -1);
    expectedSrcChannels[0] = 0;
    expectedSrcChannels[1] = 1;
    expectedSrcChannels[2] = 2;
    expectedSrcChannels[3] = 3;

    for(int i = 0; i < Us4OEMImpl::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getFrame(), 0);
        ASSERT_EQ(address.getChannel(), expectedSrcChannels[i]);
    }
}

#define MASK_CHANNELS(aperture, channelsToMask) \
do {                                            \
    for(auto channel: channelsToMask) {         \
        aperture[channel] = false;              \
    }\
} while(0)

#define SET_EXPECTED_MASKED_APERTURES(txAperture, rxAperture, channelsMask) \
    BitMask expectedTxAperture(txAperture); \
    MASK_CHANNELS(expectedTxAperture, channelsMask); \
    expectedTxApertures.push_back(::arrus::toBitset<N_ADDR_CHANNELS>(expectedTxAperture)); \
    BitMask expectedRxAperture(rxAperture); \
    MASK_CHANNELS(expectedRxAperture, channelsMask); \
    expectedRxApertures.push_back(::arrus::toBitset<N_ADDR_CHANNELS>(expectedRxAperture));

TEST_F(Us4OEMImplEsaote3ChannelsMaskTest, MasksProperlyASingleChannelForAllOperations) {
    std::unordered_set<uint8> channelsMask {7, 60, 93};
    auto us4oem = createHandle(channelsMask);
    std::vector<TxRxParameters> seq;

    const auto N_ADDR_CHANNELS = Us4OEMImpl::N_ADDR_CHANNELS;

    std::vector<BitMask> txApertures;
    std::vector<BitMask> rxApertures;
    std::vector<std::bitset<N_ADDR_CHANNELS>> expectedTxApertures;
    std::vector<std::bitset<N_ADDR_CHANNELS>> expectedRxApertures;

    {
        // Op 0:
        // Given
        BitMask txAperture(128, false);
        BitMask rxAperture(128, false);

        txAperture[0] = txAperture[7] = txAperture[33] = txAperture[95] = true;
        rxAperture[0] = rxAperture[7] = rxAperture[31] = rxAperture[60] = true;

        txApertures.push_back(txAperture);
        rxApertures.push_back(rxAperture);
        seq.push_back(
            ARRUS_STRUCT_INIT_LIST(
                TestTxRxParams,
                (
                    x.rxAperture = rxAperture,
                    x.txAperture = txAperture
                ))
                .getTxRxParameters());

        // Expected:
        SET_EXPECTED_MASKED_APERTURES(txAperture, rxAperture, channelsMask);
    }
    {
        // Op 1:
        BitMask rxAperture(128, false);
        BitMask txAperture(128, false);
        std::vector<float> txDelays(128, 0.0);

        ::arrus::setValuesInRange(txAperture, 16, 64+16, true);
        ::arrus::setValuesInRange(rxAperture, 48, 48+32, true);

        txApertures.push_back(txAperture);
        rxApertures.push_back(rxAperture);

        seq.push_back(
            ARRUS_STRUCT_INIT_LIST(
                TestTxRxParams,
                (
                    x.rxAperture = rxAperture,
                    x.txAperture = txAperture,
                    x.txDelays = txDelays
                ))
                .getTxRxParameters());
        // Expected:
        SET_EXPECTED_MASKED_APERTURES(txAperture, rxAperture, channelsMask);
    }
    {
        // Op 2:
        BitMask rxAperture(128, false);
        BitMask txAperture(128, false);
        std::vector<float> txDelays(128, 0.0);

        ::arrus::setValuesInRange(txAperture, 0, 64, true);
        ::arrus::setValuesInRange(rxAperture, 0, 32, true);

        txApertures.push_back(txAperture);
        rxApertures.push_back(rxAperture);

        seq.push_back(
            ARRUS_STRUCT_INIT_LIST(
                TestTxRxParams,
                (
                    x.rxAperture = rxAperture,
                    x.txAperture = txAperture,
                    x.txDelays = txDelays
                ))
                .getTxRxParameters());
        // Expected:
        SET_EXPECTED_MASKED_APERTURES(txAperture, rxAperture, channelsMask);
    }

    ASSERT_EQ(expectedRxApertures.size(), expectedTxApertures.size());

    for(int i = 0; i < expectedRxApertures.size(); ++i) {
        EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxApertures[i], i));
        EXPECT_CALL(*ius4oemPtr, SetTxAperture(expectedTxApertures[i], i));
    }

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(us4oem, seq);

    // Validate generated FCM
    ASSERT_EQ(fcm->getNumberOfLogicalFrames(), 3);
    ASSERT_EQ(fcm->getNumberOfLogicalChannels(), Us4OEMImpl::N_RX_CHANNELS);

    {
        // Frame 0

        std::vector<int8> expectedSrcChannels(Us4OEMImpl::N_RX_CHANNELS, -1);
        expectedSrcChannels[0] = 0;
        expectedSrcChannels[1] = 1;
        // rx aperture channel 1 is turned off (channel 7), but still we want to have it here
        expectedSrcChannels[2] = 2;
        // rx aperture channel 3 is missing (channel 60)
        expectedSrcChannels[3] = 3;

        for(int i = 0; i < Us4OEMImpl::N_RX_CHANNELS; ++i) {
            auto address = fcm->getLogical(0, i);
            EXPECT_EQ(address.getUs4oem(), 0);
            EXPECT_EQ(address.getFrame(), 0);
            ASSERT_EQ(address.getChannel(), expectedSrcChannels[i]);
        }
    }
    {
        // Frame 1, 2
        for(int frame = 1; frame <= 2; ++frame) {
            uint8 i = 0;
            ChannelIdx rxChannelNumber = 0;
            for(auto bit : rxApertures[frame]) {
                if(bit) {
                    auto address = fcm->getLogical(frame, i);
                    ASSERT_EQ(address.getUs4oem(), 0);
                    ASSERT_EQ(address.getFrame(), frame);
                    ASSERT_EQ(address.getChannel(), i++);
                }
                ++rxChannelNumber;
            }
        }
    }
}

// TX/RX reprogramming tests
class Us4OEMImplEsaote3ReprogrammingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        ius4oemPtr = dynamic_cast<MockIUs4OEM *>(ius4oem.get());
        ON_CALL(*ius4oemPtr, GetMaxTxFrequency).WillByDefault(testing::Return(MAX_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetMinTxFrequency).WillByDefault(testing::Return(MIN_TX_FREQUENCY));
    }

    Us4OEMImpl::Handle createHandle(Us4OEMSettings::ReprogrammingMode reprogrammingMode) {
        // This function can be called only once.

        BitMask activeChannelGroups = {true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true,
                                       true, true, true, true};

        std::vector<uint8> channelMapping = getRange<uint8>(0, 128);

        RxSettings rxSettings(std::nullopt, DEFAULT_PGA_GAIN, DEFAULT_LNA_GAIN, {}, 15'000'000, std::nullopt, true);

        return std::make_unique<Us4OEMImpl>(
                DeviceId(DeviceType::Us4OEM, 0),
                // NOTE: due to the below move this function can be called only once
                std::move(ius4oem), activeChannelGroups,
                channelMapping, rxSettings,
                std::unordered_set<uint8>({}),
                reprogrammingMode,
                false,
                false
        );

    }

    std::unique_ptr<IUs4OEM> ius4oem;
    MockIUs4OEM *ius4oemPtr;
    TGCCurve defaultTGCCurve;
    uint16 defaultRxBufferSize = 1;
    uint16 defaultBatchSize = 1;
    std::optional<float> defaultSri = std::nullopt;
    arrus::ops::us4r::Scheme::WorkMode defaultWorkMode = arrus::ops::us4r::Scheme::WorkMode::SYNC;
};

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, RejectsToShortPRIForSequential) {
    auto us4oem = createHandle(Us4OEMSettings::ReprogrammingMode::SEQUENTIAL);

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time + reprogramming time -1us [s]
                            // Assuming
                            x.pri = 63e-6f
                                + Us4OEMImpl::SEQUENCER_REPROGRAMMING_TIME
                                + Us4OEMImpl::RX_TIME_EPSILON
                                - 1e-6f,
                            x.sampleRange = {0, 4096}
                    ))
                    .getTxRxParameters()
    };

    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq), IllegalArgumentException);
}

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, AcceptsPriCloseTxRxTimeSequential) {
    auto us4oem = createHandle(Us4OEMSettings::ReprogrammingMode::SEQUENTIAL);

    float pri = 63e-6f + Us4OEMImpl::SEQUENCER_REPROGRAMMING_TIME
        + Us4OEMImpl::RX_TIME_EPSILON;

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time + reprogramming time [s]
                            x.pri = pri,
                            x.sampleRange = {0, 4032}
                    ))
                    .getTxRxParameters()
    };

    EXPECT_NO_THROW(SET_TX_RX_SEQUENCE(us4oem, seq));
}

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, AcceptsPriCloseTxRxTimeParallel) {
    auto us4oem = createHandle(Us4OEMSettings::ReprogrammingMode::PARALLEL);

    float pri = 63e-6f + Us4OEMImpl::RX_TIME_EPSILON;

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time only
                            x.pri = pri,
                            x.sampleRange = {0, 4032}
                    ))
                    .getTxRxParameters()
    };

    EXPECT_NO_THROW(SET_TX_RX_SEQUENCE(us4oem, seq));
}

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, RejectsToSmallPriParallel) {
    auto us4oem = createHandle(Us4OEMSettings::ReprogrammingMode::PARALLEL);

    float pri = 63e-6f-1e-6f + Us4OEMImpl::RX_TIME_EPSILON;

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time only
                            x.pri = pri,
                            x.sampleRange = {0, 4096}
                    ))
                    .getTxRxParameters()
    };

    EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq), IllegalArgumentException);
}
}

int main(int argc, char **argv) {
    std::cerr << "Starting" << std::endl;
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
