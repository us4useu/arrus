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
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;
using namespace arrus::devices::us4r;
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

class Us4OEMImplTest: public ::testing::Test {
protected:

    void SetUp() override {
        ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        ius4oemPtr = dynamic_cast<MockIUs4OEM *>(ius4oem.get());
        // Default values returned by us4oem.
        ON_CALL(*ius4oemPtr, GetMaxTxFrequency).WillByDefault(testing::Return(MAX_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetMinTxFrequency).WillByDefault(testing::Return(MIN_TX_FREQUENCY));
        ON_CALL(*ius4oemPtr, GetTxOffset).WillByDefault(testing::Return(TX_OFFSET));
    }

    Us4OEMUploadResult upload(const TxParametersSequenceColl &sequences) {
        return us4oem-> upload(sequences, defaultRxBufferSize, defaultWorkMode);
    }

    Us4OEMUploadResult upload(const TxRxParametersSequence &seq) {
        TxParametersSequenceColl seqs = {seq};
        return upload(seqs);
    }

    Us4OEMUploadResult upload(const std::vector<TxRxParameters> &txrxs) {
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = txrxs)).get();
        return upload(seq);
    }

    Us4OEMUploadResult upload(const std::vector<TxRxParameters> &txrxs, TGCCurve tgc) {
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = txrxs, x.tgcCurve = tgc)).get();
        return upload(seq);
    }

    std::unique_ptr<IUs4OEM> ius4oem;
    // A raw pointer stored to the object stored in the above unique_ptr.
    MockIUs4OEM *ius4oemPtr;
    Us4OEMImpl::Handle us4oem;
    const TGCCurve defaultTGCCurve;
    const uint16 defaultRxBufferSize = 1;
    const uint16 defaultBatchSize = 1;
    const Scheme::WorkMode defaultWorkMode = arrus::ops::us4r::Scheme::WorkMode::MANUAL;
    const std::optional<float> defaultSri = std::nullopt;
    const Us4OEMDescriptor defaultDescriptor = DEFAULT_DESCRIPTOR;
    const RxSettings defaultRxSettings{std::nullopt, DEFAULT_PGA_GAIN, DEFAULT_LNA_GAIN, {}, 15'000'000, std::nullopt, true};
};


class Us4OEMImplEsaote3LikeTest : public Us4OEMImplTest {
protected:
    void SetUp() override {
        Us4OEMImplTest::SetUp();

        std::vector<uint8> channelMapping = getRange<uint8>(0, 128);
        us4oem = std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            std::move(ius4oem),
            channelMapping, defaultRxSettings,
            Us4OEMSettings::ReprogrammingMode::SEQUENTIAL,
            defaultDescriptor,
            false,
            false
        );
    }
};

// ------------------------------------------ Parameter setters
TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxTimeAndDelay1) {
    // Sample range -> rx time
    // end-start / sampling frequency
    Interval<uint32> sampleRange(0, 1024);
    float rxDelay = 0.0f;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.sampleRange = sampleRange, x.rxDelay = rxDelay)
        )
        .get()
    };
    EXPECT_CALL(*ius4oemPtr, SetRxDelay(rxDelay, 0));
    uint32 nSamples = sampleRange.end() - sampleRange.start();
    float expectedRxTime = float(nSamples) / defaultDescriptor.getSamplingFrequency();
    EXPECT_CALL(*ius4oemPtr, SetRxTime(Ge(expectedRxTime), 0));
    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(0, _, nSamples, TX_OFFSET + sampleRange.start(), _, _, _));
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectRxTimeAndDelay2) {
    // Sample range -> rx time
    // end-start / sampling frequency
    Interval<uint32> sampleRange(40, 1024 + 40);
    float rxDelay = 1e-6f;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.sampleRange = sampleRange, x.rxDelay = rxDelay)
        )
        .get()
    };
    EXPECT_CALL(*ius4oemPtr, SetRxDelay(rxDelay, 0));
    uint32 nSamples = sampleRange.end() - sampleRange.start();
    float expectedRxTime = float(nSamples) / defaultDescriptor.getSamplingFrequency();
    EXPECT_CALL(*ius4oemPtr, SetRxTime(Ge(expectedRxTime), 0));
    EXPECT_CALL(*ius4oemPtr, ScheduleReceive(0, _, nSamples, TX_OFFSET + sampleRange.start(), _, _, _));
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfTxHalfPeriods) {
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(3e6f, 1.5, true))
        )
        .get()
    };
    EXPECT_CALL(*ius4oemPtr, SetTxHalfPeriods(3, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxFreqency(3e6f, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxInvert(true, 0));
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfTxHalfPeriods2) {
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(3e6, 3, false))
        )
        .get()
    };
    EXPECT_CALL(*ius4oemPtr, SetTxHalfPeriods(6, 0));
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfTxHalfPeriods3) {
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(10e6, 30.5, false))
        )
        .get()
    };
    EXPECT_CALL(*ius4oemPtr, SetTxHalfPeriods(61, 0));
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, TurnsOffTGCWhenEmpty) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().get()
    };
    EXPECT_CALL(*ius4oemPtr, TGCDisable);
    EXPECT_CALL(*ius4oemPtr, TGCEnable).Times(0);
    upload(seq, {});
}

TEST_F(Us4OEMImplEsaote3LikeTest, InterpolatesToTGCCharacteristicCorrectly) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().get()
    };
    TGCCurve tgc = {14.000f, 14.001f, 14.002f};

    EXPECT_CALL(*ius4oemPtr, TGCEnable);

    TGCCurve expectedTgc = {14.0f, 15.0f, 16.0f};
    // normalized
    for(float &i : expectedTgc) {
        i = (i - 14.0f) / 40.f;
    }
    EXPECT_CALL(*ius4oemPtr, TGCSetSamples(Pointwise(FloatNearPointwise(1e-4), expectedTgc), _));
    upload(seq, tgc);
}

TEST_F(Us4OEMImplEsaote3LikeTest, InterpolatesToTGCCharacteristicCorrectly2) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().get()
    };
    TGCCurve tgc = {14.000f, 14.0005f, 14.001f};

    EXPECT_CALL(*ius4oemPtr, TGCEnable);

    TGCCurve expectedTgc = {14.0f, 14.5f, 15.0f};
    // normalized
    for(float &i : expectedTgc) {
        i = (i - 14.0f) / 40.f;
    }
    EXPECT_CALL(*ius4oemPtr, TGCSetSamples(Pointwise(FloatNearPointwise(1e-4), expectedTgc), _));
    upload(seq, tgc);
}

TEST_F(Us4OEMImplEsaote3LikeTest, InterpolatesToTGCCharacteristicCorrectly3) {
    std::vector<TxRxParameters> seq = {
        TestTxRxParams().get()
    };
    TGCCurve tgc = {14.000f, 14.0002f, 14.0007f, 14.001f, 14.0015f};

    EXPECT_CALL(*ius4oemPtr, TGCEnable);

    TGCCurve expectedTgc = {14.0f, 14.2f, 14.7f, 15.0f, 15.5f};
    // normalized
    for(float &i : expectedTgc) {
        i = (i - 14.0f) / 40.f;
    }
    EXPECT_CALL(*ius4oemPtr, TGCSetSamples(Pointwise(FloatNearPointwise(1e-4), expectedTgc), _));
    upload(seq, tgc);
}

TEST_F(Us4OEMImplEsaote3LikeTest, TurnsOffAllChannelsForNOP) {
    std::vector<TxRxParameters> seq = {
        TxRxParameters::US4OEM_NOP
    };
    // empty
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> rxAperture, txAperture;
    // empty
    std::bitset<Us4OEMDescriptor::N_ACTIVE_CHANNEL_GROUPS> activeChannelGroup;
    EXPECT_CALL(*ius4oemPtr, SetRxAperture(rxAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxAperture(txAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetActiveChannelGroup(activeChannelGroup, 0));

    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectActiveChannelGroups) {
    BitMask rxAperture(128, false);
    BitMask txAperture(128, false);
    std::vector<float> txDelays(128, 0.0);
    txAperture[0] = txAperture[9] = txAperture[32] = txAperture[63] = true;
    rxAperture[6] = rxAperture[33] = rxAperture[34] = true;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.rxAperture = rxAperture,
                x.txAperture = txAperture,
                x.maskedChannelsTx = {0},
                x.maskedChannelsRx = {6}
            )
        )
        .get()
    };
    // groups: 1, 4, 7
    // active channel group mapping:
    // 1 -> 4
    // 4 -> 2
    // 7 -> 14
    std::bitset<Us4OEMDescriptor::N_ACTIVE_CHANNEL_GROUPS> expectedChannelGroups;
    expectedChannelGroups[4] = expectedChannelGroups[2] = expectedChannelGroups[14] = true;
    EXPECT_CALL(*ius4oemPtr, SetActiveChannelGroup(expectedChannelGroups, 0));
    upload(seq);
}

//// ------------------------------------------
/// TESTING CHANNEL MASKING
//// ------------------------------------------
TEST_F(Us4OEMImplEsaote3LikeTest, DoesNothingWithAperturesWhenNoChannelMask) {
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
                x.txDelays = txDelays,
                x.maskedChannelsTx = {},
                x.maskedChannelsRx = {}
            )
        )
        .get()
    };

    auto expectedTxAperture = ::arrus::toBitset<Us4OEMDescriptor::N_ADDR_CHANNELS>(txAperture);
    auto expectedRxAperture = ::arrus::toBitset<Us4OEMDescriptor::N_ADDR_CHANNELS>(rxAperture);
    auto &expectedTxDelays = txDelays;

    EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxAperture(expectedTxAperture, 0));
    for(int i = 0; i < expectedTxDelays.size(); ++i) {
        EXPECT_CALL(*ius4oemPtr, SetTxDelay(i, expectedTxDelays[i], 0, 0));
    }
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, MasksProperlyASingleChannel) {
    BitMask rxAperture(128, false);
    BitMask txAperture(128, false);
    std::vector<float> txDelays(128, 0.0);

    txAperture[0] = txAperture[7] = txAperture[33] = txAperture[95] = true;
    txDelays[0] = txDelays[7] = txDelays[33] = txDelays[95] = 1e-6;
    rxAperture[0] = rxAperture[10] = rxAperture[31] = rxAperture[60] = true;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.rxAperture = rxAperture,
                x.txAperture = txAperture,
                x.txDelays = txDelays,
                x.maskedChannelsTx = {7},
                x.maskedChannelsRx = {10}
            ))
            .get()
    };

    auto expectedTxAperture = ::arrus::toBitset<Us4OEMDescriptor::N_ADDR_CHANNELS>(txAperture);
    expectedTxAperture[7] = false;
    auto expectedRxAperture = ::arrus::toBitset<Us4OEMDescriptor::N_ADDR_CHANNELS>(rxAperture);
    expectedRxAperture[10] = false;

    std::vector<float> expectedTxDelays(txDelays);
    expectedTxDelays[7] = 0.0f;

    EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxAperture, 0));
    EXPECT_CALL(*ius4oemPtr, SetTxAperture(expectedTxAperture, 0));
    for(int i = 0; i < expectedTxDelays.size(); ++i) {
        EXPECT_CALL(*ius4oemPtr, SetTxDelay(i, expectedTxDelays[i], 0, 0));
    }
    auto result = upload(seq);
    auto fcm = result.getFCM(0);

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);
    ASSERT_EQ(fcm->getNumberOfLogicalChannels(), defaultDescriptor.getNRxChannels());

    std::vector<int8> expectedSrcChannels(defaultDescriptor.getNRxChannels(), -1);
    expectedSrcChannels[0] = 0;
    expectedSrcChannels[1] = 1;
    expectedSrcChannels[2] = 2;
    expectedSrcChannels[3] = 3;

    for(int i = 0; i < defaultDescriptor.getNRxChannels(); ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getFrame(), 0);
        ASSERT_EQ(address.getChannel(), expectedSrcChannels[i]);
    }
}


BitMask getMaskedApertureAsBitMask(BitMask aperture, std::unordered_set<ChannelIdx> mask) {
    for(auto channel: mask) {
        aperture[channel] = false;
    }
    return aperture;
}

std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> getMaskedAperture(const BitMask &aperture, std::unordered_set<ChannelIdx> mask) {
    return ::arrus::toBitset<Us4OEMDescriptor::N_ADDR_CHANNELS>(getMaskedApertureAsBitMask(aperture, mask));
}

TEST_F(Us4OEMImplEsaote3LikeTest, MasksProperlyASingleChannelForAllOperations) {
    std::unordered_set<ChannelIdx> txChannelsMask = {7, 60, 93};
    std::unordered_set<ChannelIdx> rxChannelsMask = {21, 31};
    std::vector<TxRxParameters> seq;

    const auto N_ADDR_CHANNELS = Us4OEMDescriptor::N_ADDR_CHANNELS;

    std::vector<BitMask> txApertures;
    std::vector<BitMask> rxApertures;
    std::vector<std::bitset<N_ADDR_CHANNELS>> expectedTxApertures;
    std::vector<std::bitset<N_ADDR_CHANNELS>> expectedRxApertures;

    {
        // Op :
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
                    x.txAperture = txAperture,
                    x.maskedChannelsTx = txChannelsMask,
                    x.maskedChannelsRx = rxChannelsMask
                )
            )
            .get());
        // Expected:
        expectedTxApertures.push_back(getMaskedAperture(txAperture, txChannelsMask));
        expectedRxApertures.push_back(getMaskedAperture(rxAperture, rxChannelsMask));
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
                    x.txDelays = txDelays,
                    x.maskedChannelsTx = txChannelsMask,
                    x.maskedChannelsRx = rxChannelsMask
                ))
                .get());
        // Expected:
        expectedTxApertures.push_back(getMaskedAperture(txAperture, txChannelsMask));
        expectedRxApertures.push_back(getMaskedAperture(rxAperture, rxChannelsMask));
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
                    x.txDelays = txDelays,
                    x.maskedChannelsTx = txChannelsMask,
                    x.maskedChannelsRx = rxChannelsMask
                ))
                .get());
        // Expected:
        expectedTxApertures.push_back(getMaskedAperture(txAperture, txChannelsMask));
        expectedRxApertures.push_back(getMaskedAperture(rxAperture, rxChannelsMask));
    }

    ASSERT_EQ(expectedRxApertures.size(), expectedTxApertures.size());

    ::testing::Sequence txApertureCallSequence;
    ::testing::Sequence rxApertureCallSequence;

    for(int i = 0; i < expectedRxApertures.size(); ++i) {
        EXPECT_CALL(*ius4oemPtr, SetRxAperture(expectedRxApertures[i], i)).InSequence(rxApertureCallSequence);
        EXPECT_CALL(*ius4oemPtr, SetTxAperture(expectedTxApertures[i], i)).InSequence(txApertureCallSequence);
    }

    auto result = upload(seq);
    auto fcm = result.getFCM(0);

    const auto nRxChannels = defaultDescriptor.getNRxChannels();

    // Validate generated FCM
    ASSERT_EQ(fcm->getNumberOfLogicalFrames(), 3);
    ASSERT_EQ(fcm->getNumberOfLogicalChannels(), nRxChannels);

    {
        // Frame 0

        std::vector<int8> expectedSrcChannels(nRxChannels, -1);
        expectedSrcChannels[0] = 0;
        expectedSrcChannels[1] = 1;
        // rx aperture channel 1 is turned off (channel 7), but still we want to have it here
        expectedSrcChannels[2] = 2;
        // rx aperture channel 3 is missing (channel 60)
        expectedSrcChannels[3] = 3;

        for(int i = 0; i < nRxChannels; ++i) {
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

// ------------------------------------------
// TX/RX REPROGRAMMING TESTS
// ------------------------------------------

class Us4OEMImplEsaote3ReprogrammingTest : public Us4OEMImplTest {
protected:
    void SetUp() override {
        Us4OEMImplTest::SetUp();
    }

    void setReprogrammingMode(Us4OEMSettings::ReprogrammingMode reprogrammingMode) {
        // This function can be called only once.
        std::vector<uint8> channelMapping = getRange<uint8>(0, 128);

        us4oem = std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, 0),
            // NOTE: due to the below move THIS FUNCTION CAN BE CALLED ONLY ONCE PER TEST.
            std::move(ius4oem),
            channelMapping, defaultRxSettings,
            reprogrammingMode,
            defaultDescriptor,
            false, false
        );

    }
};

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, RejectsToShortPRIForSequential) {
    setReprogrammingMode(Us4OEMSettings::ReprogrammingMode::SEQUENTIAL);

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time + reprogramming time -1us [s]
                            // Assuming
                            x.pri = 63e-6f
                                + defaultDescriptor.getSequenceReprogrammingTime()
                                + defaultDescriptor.getRxTimeEpsilon()
                                - 1e-6f,
                            x.sampleRange = {0, 4096}
                    ))
                    .get()
    };
    EXPECT_THROW(upload(seq), IllegalArgumentException);
}

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, AcceptsPriCloseTxRxTimeSequential) {
    setReprogrammingMode(Us4OEMSettings::ReprogrammingMode::SEQUENTIAL);

    float pri = 63e-6f + defaultDescriptor.getSequenceReprogrammingTime()
        + defaultDescriptor.getRxTimeEpsilon();

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time + reprogramming time [s]
                            x.pri = pri,
                            x.sampleRange = {0, 4032}
                    ))
                    .get()
    };
//    EXPECT_NO_THROW(upload(seq));
    upload(seq);
}

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, AcceptsPriCloseTxRxTimeParallel) {
    setReprogrammingMode(Us4OEMSettings::ReprogrammingMode::PARALLEL);

    float pri = 63e-6f + defaultDescriptor.getRxTimeEpsilon();

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time only
                            x.pri = pri,
                            x.sampleRange = {0, 4032}
                    ))
                    .get()
    };

    EXPECT_NO_THROW(upload(seq));
}

TEST_F(Us4OEMImplEsaote3ReprogrammingTest, RejectsToSmallPriParallel) {
    setReprogrammingMode(Us4OEMSettings::ReprogrammingMode::PARALLEL);

    float pri = 63e-6f-1e-6f + defaultDescriptor.getRxTimeEpsilon();

    std::vector<TxRxParameters> seq = {
            ARRUS_STRUCT_INIT_LIST(
                    TestTxRxParams,
                    (
                            // acquisition time only
                            x.pri = pri,
                            x.sampleRange = {0, 4096}
                    ))
                    .get()
    };

    EXPECT_THROW(upload(seq), IllegalArgumentException);
}
}

int main(int argc, char **argv) {
    std::cerr << "Starting" << std::endl;
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}