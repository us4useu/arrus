#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <utility>

#include "Us4OEMRxMappingRegisterBuilder.h"
#include "Us4OEMDescriptorFactory.h"
#include <set>
#include "arrus/core/common/tests.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {

using namespace arrus::devices;
using namespace arrus::devices::us4r;

class RXMappingTest: public ::testing::Test {
protected:

    void SetUp() override {
    }

    Us4OEMRxMappingRegister build(TxParametersSequenceColl sequences, const std::vector<uint8> channelMapping) {
        Us4OEMRxMappingRegisterBuilder builder(0, false, channelMapping, defaultDescriptor.getNRxChannels());
        builder.add(sequences);
        return builder.build();
    }

    Us4OEMRxMappingRegister build(TxParametersSequenceColl sequences) {
        return build(std::move(sequences), defaultChannelMapping);
    }

    Us4OEMRxMappingRegister build(TxRxParametersSequence sequence) {
        TxParametersSequenceColl sequences = {std::move(sequence)};
        return build(std::move(sequences));
    }

    Us4OEMRxMappingRegister build(std::vector<TxRxParameters> params) {
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = params)).get();
        return build(seq);
    }
    Us4OEMRxMappingRegister build(std::vector<TxRxParameters> params, const std::vector<uint8> channelMapping) {
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = params)).get();
        TxParametersSequenceColl sequences = {std::move(seq)};
        return build(sequences, channelMapping);
    }

    std::vector<uint8> defaultChannelMapping = getRange<uint8>(0, 128);
    const Us4OEMDescriptor defaultDescriptor = DEFAULT_DESCRIPTOR;
};


TEST_F(RXMappingTest, SetsCorrectRxMapping032) {
    // Rx aperture 0-32
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        ).get()
    };
    std::vector<uint8> expectedRxMapping = getRange<uint8>(0, 32);

    // call
    auto reg = build(seq);

    // verify
    EXPECT_EQ(reg.getMappings().size(), 1);
    auto mapping = reg.getMap(0, 0);
    EXPECT_EQ(mapping, expectedRxMapping);
}

TEST_F(RXMappingTest, SetsCorrectRxMapping032Missing1518) {
    // Rx aperture 0-32
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);
    rxAperture[15] = false;
    rxAperture[18] = false;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        ).get()
    };
    std::vector<uint8> expectedRxMapping = getRange<uint8>(0, 32);
    // 0, 1, 2, .., 14, 16, 17, 19, 20, ..., 29, 15, 18
    setValuesInRange<uint8>(expectedRxMapping, 0, 15, [](size_t i) { return (uint8) (i); });
    setValuesInRange<uint8>(expectedRxMapping, 15, 17, [](size_t i) { return (uint8) (i + 1); });
    setValuesInRange<uint8>(expectedRxMapping, 17, 30, [](size_t i) { return (uint8) (i + 2); });
    expectedRxMapping[30] = 15;
    expectedRxMapping[31] = 18;

    // call
    auto reg = build(seq);

    // verify
    EXPECT_EQ(reg.getMappings().size(), 1);
    auto mapping = reg.getMap(0, 0);
    EXPECT_EQ(mapping, expectedRxMapping);
}

TEST_F(RXMappingTest, SetsCorrectRxMapping1648) {
    // Rx aperture 0-32
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 16, 48, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        ).get()
    };
    std::vector<uint8> expectedRxMapping(32, 0);
    setValuesInRange<uint8>(expectedRxMapping, 0, 16, [](size_t i) { return static_cast<uint8>(i + 16); });
    setValuesInRange<uint8>(expectedRxMapping, 16, 32, [](size_t i) { return static_cast<uint8>(i % 16); });

    // call
    auto reg = build(seq);

    // verify
    EXPECT_EQ(reg.getMappings().size(), 1);
    auto mapping = reg.getMap(0, 0);
    EXPECT_EQ(mapping, expectedRxMapping);
}

TEST_F(RXMappingTest, SetsCorrectNumberOfMappings) {
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
            (x.rxAperture = rxAperture1)
        ).get(),
        // 2nd tx/rx
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture2)
        ).get(),
        // 3rd tx/rx
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture3)
        ).get()
    };
    std::vector<uint8> expectedRxMapping1 = getRange<uint8>(0, 32);
    std::vector<uint8> expectedRxMapping2(32, 0);
    setValuesInRange<uint8>(expectedRxMapping2, 0, 16, [](size_t i) { return static_cast<uint8>(i + 16); });
    setValuesInRange<uint8>(expectedRxMapping2, 16, 32, [](size_t i) { return static_cast<uint8>(i % 16); });

    // call
    auto reg = build(seq);

    // verify
    EXPECT_EQ(reg.getMappings().size(), 2);

    EXPECT_EQ(reg.getMapId(0, 0), 0);
    EXPECT_EQ(reg.getMapId(0, 1), 1);
    EXPECT_EQ(reg.getMapId(0, 2), 0);

    EXPECT_EQ(reg.getMap(0, 0), expectedRxMapping1);
    EXPECT_EQ(reg.getMap(0, 1), expectedRxMapping2);
}

TEST_F(RXMappingTest, TestFrameChannelMappingForNonconflictingRxMapping) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        ).get()
    };

    // call
    auto reg = build(seq);
    auto fcms = reg.acquireFCMs();
    EXPECT_EQ(fcms.size(), 1);
    const auto &fcm = fcms.at(0);
    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);
    for (size_t i = 0; i < defaultDescriptor.getNRxChannels(); ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), i);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

TEST_F(RXMappingTest, TestFrameChannelMappingForNonconflictingRxMapping2) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 16, 48, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        ).get()};

    // call
    auto reg = build(seq);
    auto fcms = reg.acquireFCMs();
    EXPECT_EQ(fcms.size(), 1);
    const auto &fcm = fcms.at(0);
    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);
    for (size_t i = 0; i < defaultDescriptor.getNRxChannels(); ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), i);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

TEST_F(RXMappingTest, TestFrameChannelMappingIncompleteRxAperture) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 32, true);

    rxAperture[31] = rxAperture[15] = false;

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        ).get()
    };
    auto reg = build(seq);
    auto fcms = reg.acquireFCMs();
    EXPECT_EQ(fcms.size(), 1);
    const auto &fcm = fcms.at(0);

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);

    for (size_t i = 0; i < 30; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), i);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

const std::vector<uint8> CONFLICTING_CHANNELS_CHANNEL_MAPPING = castTo<uint8, uint32>(
    {26,  27,  25,  23,  28,  22,  20,  21,  24,  18,  19,  15,  17,  16,  29,  13,  11,  14,  30,
     8,   12,  5,   10,  9,   31,  7,   3,   6,   0,   2,   4,   1,   56,  55,  54,  53,  57,  52,
     51,  49,  50,  48,  47,  46,  44,  45,  58,  42,  43,  59,  40,  41,  60,  38,  61,  39,  62,
     34,  37,  63,  36,  35,  32,  33,  92,  93,  89,  91,  88,  90,  87,  85,  86,  84,  83,  82,
     81,  80,  79,  77,  78,  76,  95,  75,  74,  94,  73,  72,  70,  64,  71,  68,  65,  69,  67,
     66,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
     114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127});


TEST_F(RXMappingTest, TurnsOffConflictingChannels) {
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
            (x.rxAperture = rxAperture)
        ).get()
    };

    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> expectedRxAperture;
    setValuesInRange(expectedRxAperture, 16, 48, true);
    expectedRxAperture[43] = false;
    expectedRxAperture[44] = false;
    expectedRxAperture[47] = false;

    // The channel mapping should stay unmodified
    // 27, 28, 29 are not used (should be turned off)
    std::vector<uint8> expectedRxMapping = {11, 14, 30, 8,  12, 5,  10, 9,  31, 7,  3,  6,  0,  2,  4,  1,
                                            24, 23, 22, 21, 25, 20, 19, 17, 18, 16, 15, 27, 28, 13, 26, 29};
    auto reg = build(seq, CONFLICTING_CHANNELS_CHANNEL_MAPPING);
    // verify
    EXPECT_EQ(reg.getMappings().size(), 1);
    EXPECT_EQ(reg.getMap(0, 0), expectedRxMapping);
}

TEST_F(RXMappingTest, TestFrameChannelMappingForConflictingMapping) {
    BitMask rxAperture(128, false);
    // (11, 14, 30,  8, 12,  5, 10,  9,
    //  31,  7,  3,  6,  0,  2,  4,  1,
    //  24, 23, 22, 21, 25, 20, 19, 17,
    //  18, 16, 15, 14, 12, 13, 26, 10)
    setValuesInRange(rxAperture, 16, 48, true);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams, (x.rxAperture = rxAperture)
        ).get()
    };

    auto reg = build(seq, CONFLICTING_CHANNELS_CHANNEL_MAPPING);
    auto fcms = reg.acquireFCMs();
    const auto &fcm = fcms.at(0);

    for (size_t i = 0; i < Us4OEMDescriptor::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        std::cerr << (int16) address.getChannel() << ", ";
    }
    std::cerr << std::endl;

    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 1);
    // turned off channels should be zeroed, so we just expect 0-31 here
    std::vector<int8> expectedDstChannels = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    for (size_t i = 0; i < Us4OEMDescriptor::N_RX_CHANNELS; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(address.getUs4oem(), 0);
        EXPECT_EQ(address.getChannel(), expectedDstChannels[i]);
        EXPECT_EQ(address.getFrame(), 0);
    }
}

// Interleaved RX (AFE58JD32, us4OEM+ 64): the board RX wiring table (Us4OEMDescriptorFactory).
class RXMappingInterleavedTest: public ::testing::Test {
protected:
    Us4OEMRxMappingRegister build(const BitMask &rxAperture, std::unordered_set<ChannelIdx> maskedChannelsRx = {}) {
        std::vector<TxRxParameters> params = {
            ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                   (x.rxAperture = rxAperture, x.maskedChannelsRx = maskedChannelsRx)).get()};
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = params)).get();
        TxParametersSequenceColl sequences = {std::move(seq)};
        Us4OEMRxMappingRegisterBuilder builder(0, false, getRange<uint8>(0, 128), 32, 2, table);
        builder.add(sequences);
        return builder.build();
    }
    RxInputTable table = Us4OEMDescriptorFactory::createAfe58jd32RxInputTable();
};

TEST_F(RXMappingInterleavedTest, TableIsOneToOnePerMuxGroup) {
    ASSERT_EQ(table.size(), 128);
    for (ChannelIdx group = 0; group < 2; ++group) {
        std::set<std::pair<uint8_t, uint8_t>> locations;
        for (ChannelIdx ch = group * 64; ch < (group + 1) * 64; ++ch) {
            EXPECT_LT(table[ch].first, 32);
            EXPECT_LT(table[ch].second, 2);
            locations.insert(table[ch]);
        }
        EXPECT_EQ(locations.size(), 64);
    }
}

TEST_F(RXMappingInterleavedTest, TableMatchesSchematic) {
    // AFE on slots 0-15: INP1 <- RX_IN16 = LVOUT32 (odd), INP2 <- RX_IN0 = LVOUT0 (even).
    EXPECT_EQ(table[32], (std::pair<uint8_t, uint8_t>{0, 0}));
    EXPECT_EQ(table[0], (std::pair<uint8_t, uint8_t>{0, 1}));
    // INP25 <- RX_IN14 = LVOUT14, INP26 <- RX_IN30 = LVOUT46 (converter 12).
    EXPECT_EQ(table[14], (std::pair<uint8_t, uint8_t>{12, 0}));
    EXPECT_EQ(table[46], (std::pair<uint8_t, uint8_t>{12, 1}));
    // AFE on slots 16-31: the same pattern + 16.
    EXPECT_EQ(table[48], (std::pair<uint8_t, uint8_t>{16, 0}));
    EXPECT_EQ(table[16], (std::pair<uint8_t, uint8_t>{16, 1}));
    // The second mux group (transducer channels 64-127): the same location as c - 64.
    for (ChannelIdx ch = 64; ch < 128; ++ch) {
        EXPECT_EQ(table[ch], table[ch - 64]);
    }
}

TEST_F(RXMappingInterleavedTest, FullMuxGroupInSingleRx) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 64, true);
    auto reg = build(rxAperture);
    EXPECT_EQ(reg.getMappings().size(), 1);
    auto map = reg.getMap(0, 0);
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> expectedRxAperture;
    setValuesInRange(expectedRxAperture, 0, 64, true);
    EXPECT_EQ(reg.getRxAperture(0, 0), expectedRxAperture);
    auto fcms = reg.acquireFCMs();
    const auto &fcm = fcms.at(0);
    std::set<int8> columns;
    for (ChannelIdx ch = 0; ch < 64; ++ch) {
        auto column = fcm->getLogical(0, ch).getChannel();
        columns.insert(column);
        // Column = raw row * 32 + mapping entry selecting the channel's RX slot.
        EXPECT_EQ(column / 32, table[ch].second) << "channel " << ch;
        EXPECT_EQ(map.at(column % 32), table[ch].first) << "channel " << ch;
    }
    EXPECT_EQ(columns.size(), 64);
}

TEST_F(RXMappingInterleavedTest, DifferentMuxGroupOfTheSameSlotConflicts) {
    BitMask rxAperture(128, false);
    rxAperture[0] = rxAperture[32] = rxAperture[64] = true; // 0, 32: slot 0 (rows 1, 0); 64: slot 0, other mux group
    auto reg = build(rxAperture);
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> expectedRxAperture;
    expectedRxAperture[0] = expectedRxAperture[32] = true;
    EXPECT_EQ(reg.getRxAperture(0, 0), expectedRxAperture);
    auto fcms = reg.acquireFCMs();
    const auto &fcm = fcms.at(0);
    EXPECT_EQ(fcm->getLogical(0, 0).getChannel(), 32); // channel 0: slot 0, second raw row
    EXPECT_EQ(fcm->getLogical(0, 1).getChannel(), 0);  // channel 32: slot 0, first raw row
}

TEST_F(RXMappingInterleavedTest, MaskedChannelDoesNotDropItsPair) {
    BitMask rxAperture(128, false);
    rxAperture[0] = rxAperture[32] = true;
    auto reg = build(rxAperture, {0});
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> expectedRxAperture;
    expectedRxAperture[32] = true;
    EXPECT_EQ(reg.getRxAperture(0, 0), expectedRxAperture);
    auto map = reg.getMap(0, 0);
    auto fcms = reg.acquireFCMs();
    const auto &fcm = fcms.at(0);
    auto column = fcm->getLogical(0, 1).getChannel();
    // The pair (channel 32) must be read from the first raw row of the column selecting RX slot 0.
    ASSERT_LT(column, 32);
    EXPECT_EQ(map.at(column), 0);
}

TEST_F(RXMappingInterleavedTest, FullMuxGroupWithMaskedChannels) {
    BitMask rxAperture(128, false);
    setValuesInRange(rxAperture, 0, 64, true);
    // 0 and 32 share RX slot 0 (masked channel first); 5 is in slot 5 with channel 7 (masked channel second).
    auto reg = build(rxAperture, {0, 7});
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> expectedRxAperture;
    setValuesInRange(expectedRxAperture, 0, 64, true);
    expectedRxAperture[0] = expectedRxAperture[7] = false;
    EXPECT_EQ(reg.getRxAperture(0, 0), expectedRxAperture);
    auto map = reg.getMap(0, 0);
    auto fcms = reg.acquireFCMs();
    const auto &fcm = fcms.at(0);
    for (ChannelIdx ch = 0; ch < 64; ++ch) {
        auto column = fcm->getLogical(0, ch).getChannel();
        EXPECT_EQ(column / 32, table[ch].second) << "channel " << ch;
        EXPECT_EQ(map.at(column % 32), table[ch].first) << "channel " << ch;
    }
}
}// namespace


int main(int argc, char **argv) {
    std::cerr << "Starting" << std::endl;
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
