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
};


#define SET_TX_RX_SEQUENCE_TGC(us4oem, seq, tgc) \
     us4oem->setTxRxSequence(seq, tgc, defaultRxBufferSize, defaultBatchSize, defaultSri)

#define SET_TX_RX_SEQUENCE(us4oem, seq) SET_TX_RX_SEQUENCE_TGC(us4oem, seq, defaultTGCCurve)

// ------------------------------------------ Testing parameters set to IUs4OEM
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