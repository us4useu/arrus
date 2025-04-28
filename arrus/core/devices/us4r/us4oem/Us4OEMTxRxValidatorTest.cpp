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
    float invalidTxDelay = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx2().getDelay().end();
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
    float invalidTxDelay = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx2().getDelay().start();
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

TEST_F(Us4OEMTxRxValidatorTest, PreventsInvalidAmplitudeLevel) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(2e6, 2.0f, false, 0))
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

TEST_F(Us4OEMTxRxValidatorTest, PreventsToLongPulse) {
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(10e6, 33.0f, false))
        ).get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST_F(Us4OEMTxRxValidatorTest, PreventsTooHighFrequency) {
    const auto maxFreq = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx2().getFrequency().end();
    Pulse pulse(maxFreq+1e5, 1.0f, false);
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse =  pulse)
        )
        .get()
    };
    TxRxParametersSequence seq = getSequence(txrxs);
    EXPECT_THROW(validate(seq), IllegalArgumentException);

}
TEST_F(Us4OEMTxRxValidatorTest, PreventsTooLowFrequency) {
    const auto minFreq = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx2().getFrequency().start();
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

TEST_F(Us4OEMTxRxValidatorTest, PreventsTooLongPulse) {
    const auto maxCycles = DEFAULT_DESCRIPTOR.getTxRxSequenceLimits().getTxRx().getTx2().getPulseCycles().end();
    const float maxNCycles = maxCycles;
    std::vector<TxRxParameters> txrxs = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (x.pulse = Pulse(5e6, maxNCycles+1.0f, false))
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

// TODO multiple ops to verify
// TODO PreventsToManyOps
int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
