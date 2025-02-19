#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ius4oem.h>

#include "arrus/core/common/tests.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactoryImpl.h"

#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"
#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/core/common/logging.h"

namespace {

using namespace arrus;
using namespace arrus::devices;

using ::testing::_;
using ::testing::FloatEq;
using ::testing::Eq;
using ::testing::Pointwise;
using ::testing::InSequence;
using ::testing::Lt;
using ::testing::Gt;

// Below default parameters should be conformant with CommonSettings.h
//struct ExpectedUs4RParameters {
//    PGA_GAIN pgaGain = PGA_GAIN::PGA_GAIN_30dB;
//    LNA_GAIN_GBL lnaGain = LNA_GAIN_GBL::LNA_GAIN_GBL_24dB;
//    DIG_TGC_ATTENUATION dtgcAttValue = DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_42dB;
//    EN_DIG_TGC dtgcAttEnabled = EN_DIG_TGC::EN_DIG_TGC_EN;
//    RxSettings::TGCCurve tgcSamplesNormalized = getRange<float>(0.4, 0.65, 0.0125);
//    bool tgcEnabled = true;
//    LPF_PROG lpfCutoff = LPF_PROG::LPF_PROG_10MHz;
//    GBL_ACTIVE_TERM activeTerminationValue = GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50;
//    ACTIVE_TERM_EN activeTerminationEnabled = ACTIVE_TERM_EN::ACTIVE_TERM_EN;
//};
//
std::optional<Us4RTxRxLimits> DEFAULT_US4R_LIMITS = std::nullopt;

//class Us4OEMFactorySimpleParametersTest
//        : public testing::TestWithParam<std::pair<TestUs4OEMSettings, ExpectedUs4RParameters>> {
//protected:
//    std::optional<Us4RTxRxLimits> defaultUs4RLimits = std::nullopt;
//};

//TEST_P(Us4OEMFactorySimpleParametersTest, VerifyUs4OEMFactorySimpleParameters) {
//    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
//    ExpectedUs4RParameters us4rParameters = GetParam().second;
//    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));
//    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetPGAGain(us4rParameters.pgaGain));
//    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetLNAGain(us4rParameters.lnaGain));
//    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetDTGC(us4rParameters.dtgcAttEnabled, us4rParameters.dtgcAttValue));
//    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetLPFCutoff(us4rParameters.lpfCutoff));
//    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetActiveTermination(us4rParameters.activeTerminationEnabled,
//                                       us4rParameters.activeTerminationValue));
//    Us4OEMFactoryImpl factory;
//    factory.getUs4OEM(0, ius4oem, GetParam().first.getUs4OEMSettings(), false, false, defaultUs4RLimits);
//}
//
//INSTANTIATE_TEST_CASE_P
//
//(Us4OEMFactorySetsSimpleIUs4OEMProperties, Us4OEMFactorySimpleParametersTest,
// testing::Values(
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.dtgcAttenuation=42, x.tgcSamples={})), ExpectedUs4RParameters{}},
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.pgaGain=24, x.dtgcAttenuation=42, x.tgcSamples={})),
//                   ARRUS_STRUCT_INIT_LIST(ExpectedUs4RParameters, (x.pgaGain=PGA_GAIN::PGA_GAIN_24dB))},
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.lnaGain=12, x.dtgcAttenuation=42, x.tgcSamples={})),
//                   ARRUS_STRUCT_INIT_LIST(ExpectedUs4RParameters, (x.lnaGain=LNA_GAIN_GBL::LNA_GAIN_GBL_12dB))},
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.lpfCutoff=(int)15e6, x.dtgcAttenuation=42, x.tgcSamples={})),
//                   ARRUS_STRUCT_INIT_LIST(ExpectedUs4RParameters, (x.lpfCutoff=LPF_PROG::LPF_PROG_15MHz))}
// ));
//
//class Us4OEMFactoryOptionalParametersTest
//        : public testing::TestWithParam<std::pair<TestUs4OEMSettings, ExpectedUs4RParameters>> {
//protected:
//    std::optional<Us4RTxRxLimits> defaultUs4RLimits = std::nullopt;
//};
//
//TEST_P(Us4OEMFactoryOptionalParametersTest, VerifyUs4OEMFactoryOptionalParameters) {
//    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
//    ExpectedUs4RParameters us4rParameters = GetParam().second;
//    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));
//    if(GetParam().first.dtgcAttenuation.has_value()) {
//        // NO disable
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetDTGC(EN_DIG_TGC::EN_DIG_TGC_DIS, _)).Times(0);
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetDTGC(EN_DIG_TGC::EN_DIG_TGC_EN, us4rParameters.dtgcAttValue));
//    } else {
//        // NO enable
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetDTGC(EN_DIG_TGC::EN_DIG_TGC_EN, _)).Times(0);
//
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetDTGC(EN_DIG_TGC::EN_DIG_TGC_DIS, _));
//    }
//
//    if(GetParam().first.activeTermination.has_value()) {
//        // NO disable
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_DIS,
//                                           us4rParameters.activeTerminationValue))
//                .Times(0);
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_EN,
//                                           us4rParameters.activeTerminationValue));
//    } else {
//        // NO enable
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_EN, testing::Matcher<::us4r::afe58jd48::GBL_ACTIVE_TERM>(_))).Times(0);
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_DIS, testing::Matcher<::us4r::afe58jd48::GBL_ACTIVE_TERM>(_)));
//    }
//    Us4OEMFactoryImpl factory;
//    factory.getUs4OEM(0, ius4oem, GetParam().first.getUs4OEMSettings(), false, false, defaultUs4RLimits);
//}
//
//INSTANTIATE_TEST_CASE_P
//
//(Us4OEMFactorySetsOptionalIUs4OEMProperties,
// Us4OEMFactoryOptionalParametersTest,
// testing::Values(
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.dtgcAttenuation=0, x.tgcSamples={})),
//                   ARRUS_STRUCT_INIT_LIST(ExpectedUs4RParameters,
//                                          (x.dtgcAttValue=DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_0dB))},
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.dtgcAttenuation={})),
//                   ExpectedUs4RParameters{}}, // Any value is accepted
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.activeTermination=200, x.tgcSamples={})),
//                   ARRUS_STRUCT_INIT_LIST(ExpectedUs4RParameters,
//                                          (x.activeTerminationValue=GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_200))},
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.activeTermination={})), ExpectedUs4RParameters{}}
// ));
//
//
//class Us4OEMFactoryTGCSamplesTest : public testing::TestWithParam<std::pair<TestUs4OEMSettings, ExpectedUs4RParameters>> {
//protected:
//    std::optional<Us4RTxRxLimits> defaultUs4RLimits = std::nullopt;
//};

// TODO(ARRUS-179)
//TEST_P(Us4OEMFactoryTGCSamplesTest, VerifyUs4OEMFactorySimpleParameters) {
//    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
//    ExpectedUs4RParameters us4rParameters = GetParam().second;
//    RxSettings::TGCCurve tgcCurve = GetParam().first.getUs4OEMSettings().getRxSettings().getTgcSamples();
//    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));
//
//    if(tgcCurve.empty()) {
//        // NO TGC enable
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCEnable()).Times(0);
//    } else {
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCEnable());
//        // NO TGC disable
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCDisable()).Times(0);
//        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCSetSamples(Pointwise(FloatEq(), us4rParameters.tgcSamplesNormalized), _));
//    }
//
//    Us4OEMFactoryImpl factory;
//
//    factory.getUs4OEM(0, ius4oem, GetParam().first.getUs4OEMSettings(), false, false, defaultUs4RLimits);
//}
//
//INSTANTIATE_TEST_CASE_P
//
//(Us4OEMFactorySetsTGCSettings,
// Us4OEMFactoryTGCSamplesTest,
// testing::Values(
//         // NO TGC
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.tgcSamples={})), ExpectedUs4RParameters{}},
//         // TGC samples set
//         std::pair{ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
//                    x.pgaGain=30,
//                    x.lnaGain=24,
//                    x.tgcSamples={30, 35, 40},
//                    x.isApplyCharacteristic=false)),
//                   ARRUS_STRUCT_INIT_LIST(ExpectedUs4RParameters, (x.tgcSamplesNormalized={0.4, 0.525, 0.65}))}
// ));

// Mappings.

TEST(Us4OEMFactoryTest, WorksForConsistentMapping) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));

    // Mapping includes groups of 32 channel, each has the same permutation
    std::vector<uint8_t> channelMapping = getRange<uint8_t>(0, 128, 1);

    for(int i = 0; i < 4; ++i) {
        std::swap(channelMapping[i * 32], channelMapping[(i + 1) * 32 - 1]);
    }

    TestUs4OEMSettings cfg;
    cfg.channelMapping = std::vector<ChannelIdx>(
            std::begin(channelMapping), std::end(channelMapping));
    Us4OEMFactoryImpl factory;
    // Expect
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetRxChannelMapping(_, _)).Times(0);
    EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                SetRxChannelMapping(
                        std::vector<uint8_t>(
                                std::begin(channelMapping),
                                std::begin(channelMapping) + 32), 0))
            .Times(1);
    // Run
    factory.getUs4OEM(0, ius4oem, cfg.getUs4OEMSettings(), false, false, DEFAULT_US4R_LIMITS);
}

TEST(Us4OEMFactoryTest, WorksForInconsistentMapping) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));

    // Mapping includes groups of 32 channel, each has the same permutation
    std::vector<uint8_t> channelMapping = getRange<uint8_t>(0, 128, 1);

    for(int i = 0; i < 2; ++i) {
        std::swap(channelMapping[i * 32], channelMapping[(i + 1) * 32 - 1]);
    }

    for(int i = 2; i < 4; ++i) {
        std::swap(channelMapping[i * 32 + 1], channelMapping[(i + 1) * 32 - 2]);
    }

    TestUs4OEMSettings cfg;
    cfg.channelMapping = std::vector<ChannelIdx>(
            std::begin(channelMapping), std::end(channelMapping));
    Us4OEMFactoryImpl factory;
    // Expect
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetRxChannelMapping(_, _)).Times(0);
    // Run
    factory.getUs4OEM(0, ius4oem, cfg.getUs4OEMSettings(), false, false, DEFAULT_US4R_LIMITS);
}

// Tx channel mapping
TEST(Us4OEMFactoryTest, WorksForTxChannelMapping) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));

    std::vector<ChannelIdx> channelMapping = getRange<ChannelIdx>(0, 128, 1);
    TestUs4OEMSettings cfg;
    cfg.channelMapping = channelMapping;
    Us4OEMFactoryImpl factory;
    // Expect
    {
        InSequence seq;
        for(ChannelIdx i = 0; i < Us4OEMDescriptor::N_TX_CHANNELS; ++i) {
            EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                        SetTxChannelMapping(i, channelMapping[i]));
        }

    }
    // No other calls should be made
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetTxChannelMapping(Lt(0), _))
            .Times(0);
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetTxChannelMapping(Gt(127), _))
            .Times(0);

    // Run
    factory.getUs4OEM(0, ius4oem, cfg.getUs4OEMSettings(), false, false, DEFAULT_US4R_LIMITS);
}

TEST(Us4OEMFactoryTest, SetsAppropriateTxRxLimits) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ON_CALL(GET_MOCK_PTR(ius4oem), GetOemVersion()).WillByDefault(::testing::Return(1));

    // All parameters set
    Interval<float> pulseLengthLimits{1e-6f, 10e-6f};
    Interval<Voltage> voltageLimits{10, 40};
    Interval<float> priLimits{10e-6, 100e-6};
    Us4RTxRxLimits limits0{pulseLengthLimits, voltageLimits, priLimits};

    // Run
    auto descriptor0 = Us4OEMFactoryImpl::getOEMDescriptor(0, ius4oem, limits0);
    const auto &txRxLimits0 = descriptor0.getTxRxSequenceLimits().getTxRx();
    EXPECT_EQ(txRxLimits0.getPri(), priLimits);
    EXPECT_EQ(txRxLimits0.getTx1().getPulseLength(), pulseLengthLimits);
    EXPECT_EQ(txRxLimits0.getTx1().getVoltage(), voltageLimits);

    Us4RTxRxLimits limits1{std::nullopt, voltageLimits, priLimits};
    auto descriptor1 = Us4OEMFactoryImpl::getOEMDescriptor(0, ius4oem, limits1);
    const auto &txRxLimits1 = descriptor1.getTxRxSequenceLimits().getTxRx();
    EXPECT_EQ(txRxLimits1.getPri(), priLimits);
    EXPECT_EQ(txRxLimits1.getTx1().getPulseCycles(),  Interval<float>(0.5f, 32.0f)); // Us4OEMFactoryImpl default
    EXPECT_EQ(txRxLimits1.getTx1().getVoltage(), voltageLimits);

    Us4RTxRxLimits limits2{pulseLengthLimits, std::nullopt, priLimits};
    auto descriptor2 = Us4OEMFactoryImpl::getOEMDescriptor(0, ius4oem, limits2);
    const auto &txRxLimits2 = descriptor2.getTxRxSequenceLimits().getTxRx();
    EXPECT_EQ(txRxLimits2.getPri(), priLimits);
    EXPECT_EQ(txRxLimits2.getTx1().getPulseLength(),  pulseLengthLimits);
    EXPECT_EQ(txRxLimits2.getTx1().getVoltage(), Interval<Voltage>(5, 90)); // Us4OEMFactoryImpl default value

    Us4RTxRxLimits limits3{pulseLengthLimits, voltageLimits, std::nullopt};
    auto descriptor3 = Us4OEMFactoryImpl::getOEMDescriptor(0, ius4oem, limits3);
    const auto &txRxLimits3 = descriptor3.getTxRxSequenceLimits().getTxRx();
    EXPECT_EQ(txRxLimits3.getPri(), Interval<float>(35e-6f, 1.0f)); // Us4OEMFactoryImpl default value
    EXPECT_EQ(txRxLimits3.getTx1().getPulseLength(),  pulseLengthLimits);
    EXPECT_EQ(txRxLimits3.getTx1().getVoltage(), voltageLimits);
}
}

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


