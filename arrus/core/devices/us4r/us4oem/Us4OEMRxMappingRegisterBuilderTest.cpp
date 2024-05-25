
namespace {
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
}
