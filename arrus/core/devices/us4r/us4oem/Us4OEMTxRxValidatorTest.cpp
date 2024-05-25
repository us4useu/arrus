#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;
using ::testing::_;
using ::testing::Ge;
using ::testing::FloatEq;
using ::testing::FloatNearus
using ::testing::Pointwise;

MATCHER_P(FloatNearPointwise, tol,
"") {
return
std::abs (std::get<0>(arg)
-
std::get<1>(arg)
) <
tol;
}

constexpr uint16 DEFAULT_PGA_GAIN = 30;
constexpr uint16 DEFAULT_LNA_GAIN = 24;
constexpr float MAX_TX_FREQUENCY = 65e6f;
constexpr float MIN_TX_FREQUENCY = 1e6f;
constexpr uint32_t TX_OFFSET = 123;

struct TestTxRxParams {

    TestTxRxParams() {
        for (int i = 0; i < 32; ++i) {
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
        std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock < MockIUs4OEM>>
        ();
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

// ------------------------------------------ TESTING VALIDATION
TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidApertureSize
) {
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
for(
size_t i = 0;
i < 33; ++i) {
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

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidTxDelays
) {
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

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidPri
) {
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

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidNPeriodsOnly
) {
// Tx delays
std::vector<TxRxParameters> seq = {
    ARRUS_STRUCT_INIT_LIST(
        TestTxRxParams,
        (x.pulse = Pulse(2e6, 1.3f, false)))
        .getTxRxParameters()
};
EXPECT_THROW(SET_TX_RX_SEQUENCE(us4oem, seq),
IllegalArgumentException);

seq = {
    ARRUS_STRUCT_INIT_LIST(
        TestTxRxParams,
        (x.pulse = Pulse(2e6, 1.5f, false)))
        .getTxRxParameters()
};
SET_TX_RX_SEQUENCE(us4oem, seq);
}

TEST_F(Us4OEMImplEsaote3LikeTest, PreventsInvalidFrequency
) {
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
}
