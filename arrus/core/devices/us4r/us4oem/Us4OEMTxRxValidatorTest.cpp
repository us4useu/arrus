#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>

#include "Us4OEMImpl.h"
#include "Us4OEMTxRxValidator.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;
using ::testing::_;
using ::testing::FloatEq;
using ::testing::Ge;
using ::testing::FloatNear;
using ::testing::Pointwise;

struct TestTxRxParams {

    TestTxRxParams() {
        for (int i = 0; i < 32; ++i) {
            rxAperture[i] = true;
        }
    }

    BitMask txAperture = getNTimes(true, Us4OEMDescriptor::N_TX_CHANNELS);
    std::vector<float> txDelays = getNTimes(0.0f, Us4OEMDescriptor::N_TX_CHANNELS);
    ops::us4r::Pulse pulse{2.0e6f, 2.5f, true};
    BitMask rxAperture = getNTimes(false, Us4OEMDescriptor::N_ADDR_CHANNELS);
    uint32 decimationFactor = 1;
    float pri = 200e-6f;
    Interval<uint32> sampleRange{0, 4096};
    std::optional<BitstreamId> bitstreamId{std::nullopt};
    std::unordered_set<ChannelIdx> maskedChannelsTx = {};
    std::unordered_set<ChannelIdx> maskedChannelsRx = {};
    Tuple<ChannelIdx> rxPadding = {0, 0};
    float rxDelay = 0.0f;

    [[nodiscard]] arrus::devices::us4r::TxRxParameters get() const {
        return TxRxParameters(
            txAperture, txDelays, pulse, rxAperture, sampleRange, decimationFactor, pri,
            rxPadding, rxDelay, bitstreamId, maskedChannelsTx, maskedChannelsRx);
    }
};

struct TestTxRxParamsSequence {
    std::vector<TxRxParameters> txrx = {TestTxRxParams{}.get()};
    uint16 nRepeats = 1;
    std::optional<float> sri = std::nullopt;
    ops::us4r::TGCCurve tgcCurve = {};
    DeviceId txProbeId{arrus::devices::DeviceType::Probe, 0};
    DeviceId rxProbeId{arrus::devices::DeviceType::Probe, 0};

    [[nodiscard]] arrus::devices::us4r::TxRxParametersSequence get() const {
        return TxRxParametersSequence{
            txrx, nRepeats, sri, tgcCurve, txProbeId, rxProbeId
        };
    }
};

class Us4OEMTxRxValidatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void validate(const TxRxParametersSequence &seq) {
        Us4OEMTxRxValidator validator("test validator", DEFAULT_DESCRIPTOR, true);
        validator.validate(seq);
        validator.throwOnErrors();
    }

    TxRxParametersSequence getSequence(std::vector<TxRxParameters> &params) {
        return ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = params)).get();
    }
};

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvalidTxApertureSize) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txAperture = getNTimes(true, Us4OEMDescriptor::N_TX_CHANNELS + 1)))
            .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvalidRxApertureSize) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = getNTimes(true, Us4OEMDescriptor::N_ADDR_CHANNELS - 1))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvalidNumberOfRxActiveChannels) {
    BitMask rxAperture(128, false);
    for (size_t i = 0; i < 33; ++i) {
        rxAperture[i] = true;
    }
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.rxAperture = rxAperture)
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F (Us4OEMTxRxValidatorTest, PreventsInvalidNumberOfTxDelays) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(0.0f, Us4OEMDescriptor::N_TX_CHANNELS/2))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F (Us4OEMTxRxValidatorTest, PreventsTooLongTxDelay) {
    float invalidTxDelay = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx().getDelay().end();
    invalidTxDelay += 1e-6f;
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(invalidTxDelay, Us4OEMDescriptor::N_TX_CHANNELS))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F (Us4OEMTxRxValidatorTest, PreventsTooShortOrNegativeTxDelay) {
    float invalidTxDelay = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx().getDelay().start();
    invalidTxDelay -= 1e-6f;
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.txDelays = getNTimes(invalidTxDelay, Us4OEMDescriptor::N_TX_CHANNELS))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F (Us4OEMTxRxValidatorTest, PreventsTooLongPri) {
    float invalidPri = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getPri().end();
    invalidPri += 1e-6f;
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pri = invalidPri)
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvalidNPeriods) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 1.3f, false))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, AcceptsCorrectNPeriods) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 1.5f, false))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    validate(seq);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsTooHighFrequency) {
    const auto maxFreq = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx().getFrequency().end();
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse =  Pulse(std::nextafter(maxFreq, maxFreq + 1e6f), 1.0f, false))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);

}
TEST_F(Us4OEMTxRxValidatorTest, PreventsTooLowFrequency) {
    const auto minFreq = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx().getFrequency().start();
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(minFreq-0.5e6f, 1.0f, false))
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvaldTxChannelsMask) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.maskedChannelsTx = {128})
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvaldRxChannelsMask) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.maskedChannelsRx = {129})
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

}// namespace

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
