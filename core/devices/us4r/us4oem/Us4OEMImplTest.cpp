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
                            [](size_t i) { return (uint8) (i+1); });
    setValuesInRange<uint8>(expectedRxMapping, 17, 30,
                            [](size_t i) { return (uint8) (i+2); });
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

//TEST_F(Us4OEMImplEsaote3LikeTest, SetsCorrectNumberOfMappings) {
//    // TODO trzy operacje, dwie z nich maja taki sam rxMapping: czy powstana tylko dwa mapowania?
//
//}
//
//// TODO checks to not exceed maximum number of rx mappings
//
//// TODO Test fixture with different (conflicting) channel mapping here is required
//TEST(Us4OEMImplEsaote3LikeTest, SetsCorrectlyIrregularRxMapping) {
//
//}
//
//TEST(Us4OEMImplEsaote3LikeTest, TurnsOffConflictingChannels) {
//
//}


// TGC, interpolation to the destination values
// sample range, rx time
// active channel groups! (NOP, no NOP)
// tx half periods

// Test NOP handling

// Test generated FrameMapping

}

int main(int argc, char **argv) {
    std::cerr << "Starting" << std::endl;
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}