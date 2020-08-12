#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ius4oem.h>

#include "arrus/core/devices/us4r/us4oem/Us4OEMFactoryImpl.h"

#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {

using namespace arrus;
using namespace us4r::afe58jd18;

using ::testing::_;
using ::testing::FloatEq;
using ::testing::Eq;
using ::testing::Pointwise;
using ::testing::InSequence;
using ::testing::Lt;
using ::testing::Gt;

#define GET_MOCK_PTR(sptr) *(MockIUs4OEM *) (sptr.get())

// Below default parameters should be conformant with CommonSettings.h
struct ExpectedUs4RParameters {
    PGA_GAIN pgaGain = PGA_GAIN::PGA_GAIN_30dB;
    LNA_GAIN_GBL lnaGain = LNA_GAIN_GBL::LNA_GAIN_GBL_24dB;
    DIG_TGC_ATTENUATION dtgcAttValue = DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_42dB;
    EN_DIG_TGC dtgcAttEnabled = EN_DIG_TGC::EN_DIG_TGC_EN;
    TGCCurve tgcSamplesNormalized = getRange<float>(0.4, 0.65, 0.0125);
    bool tgcEnabled = true;
    LPF_PROG lpfCutoff = LPF_PROG::LPF_PROG_10MHz;
    GBL_ACTIVE_TERM activeTerminationValue = GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50;
    ACTIVE_TERM_EN activeTerminationEnabled = ACTIVE_TERM_EN::ACTIVE_TERM_EN;
};

// Mocks.
class MockIUs4OEM : public IUs4OEM {
public:
    MOCK_METHOD(unsigned int, GetID, (), (override));
    MOCK_METHOD(bool, IsPowereddown, (), (override));
    MOCK_METHOD(void, Initialize, (int), (override));
    MOCK_METHOD(void, Synchronize, (), (override));
    MOCK_METHOD(void, ScheduleReceive,
                (const size_t firing, const size_t address, const size_t length, const uint32_t start, const uint32_t decimation, const size_t rxMapId, const std::function<void()>& callback),
                (override));
    MOCK_METHOD(void, ClearScheduledReceive, (), (override));
    MOCK_METHOD(void, TransferRXBufferToHost,
                (unsigned char * dstAddress, size_t length, size_t srcAddress),
                (override));
    MOCK_METHOD(void, SetPGAGain, (us4r::afe58jd18::PGA_GAIN gain), (override));
    MOCK_METHOD(void, SetLPFCutoff, (us4r::afe58jd18::LPF_PROG cutoff),
                (override));
    MOCK_METHOD(void, SetActiveTermination,
                (us4r::afe58jd18::ACTIVE_TERM_EN endis, us4r::afe58jd18::GBL_ACTIVE_TERM term),
                (override));
    MOCK_METHOD(void, SetLNAGain, (us4r::afe58jd18::LNA_GAIN_GBL gain),
                (override));
    MOCK_METHOD(void, SetDTGC,
                (us4r::afe58jd18::EN_DIG_TGC endis, us4r::afe58jd18::DIG_TGC_ATTENUATION att),
                (override));
    MOCK_METHOD(void, InitializeTX, (), (override));
    MOCK_METHOD(void, SetNumberOfFirings, (const unsigned short nFirings),
                (override));
    MOCK_METHOD(float, SetTxDelay,
                (const unsigned char channel, const float value, const unsigned short firing),
                (override));
    MOCK_METHOD(float, SetTxFreqency,
                (const float frequency, const unsigned short firing),
                (override));
    MOCK_METHOD(unsigned char, SetTxHalfPeriods,
                (unsigned char nop, const unsigned short firing), (override));
    MOCK_METHOD(void, SetTxInvert, (bool onoff, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetTxCw, (bool onoff, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetRxAperture,
                (const unsigned char origin, const unsigned char size, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetTxAperture,
                (const unsigned char origin, const unsigned char size, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetRxAperture,
                (const std::bitset<NCH>& aperture, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetTxAperture,
                (const std::bitset<NCH>& aperture, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetActiveChannelGroup,
                (const std::bitset<NCH / 8>& group, const unsigned short firing),
                (override));
    MOCK_METHOD(void, SetRxTime,
                (const float time, const unsigned short firing), (override));
    MOCK_METHOD(void, SetRxDelay,
                (const float delay, const unsigned short firing), (override));
    MOCK_METHOD(void, EnableTransmit, (), (override));
    MOCK_METHOD(void, EnableSequencer, (), (override));
    MOCK_METHOD(void, SetRxChannelMapping,
                ( const std::vector<uint8_t> & mapping, const uint16_t rxMapId),
                (override));
    MOCK_METHOD(void, SetTxChannelMapping,
                (const unsigned char srcChannel, const unsigned char dstChannel),
                (override));
    MOCK_METHOD(void, TGCEnable, (), (override));
    MOCK_METHOD(void, TGCDisable, (), (override));
    MOCK_METHOD(void, TGCSetSamples,
                (const std::vector<float> & samples, const int firing),
                (override));
    MOCK_METHOD(void, TriggerStart, (), (override));
    MOCK_METHOD(void, TriggerStop, (), (override));
    MOCK_METHOD(void, TriggerSync, (), (override));
    MOCK_METHOD(void, SetNTriggers, (unsigned short n), (override));
    MOCK_METHOD(void, SetTrigger,
                (unsigned short timeToNextTrigger, bool syncReq, unsigned short idx),
                (override));
    MOCK_METHOD(void, UpdateFirmware, (const char * filename), (override));
    MOCK_METHOD(float, GetUpdateFirmwareProgress, (), (override));
    MOCK_METHOD(const char *, GetUpdateFirmwareStatus, (), (override));
    MOCK_METHOD(int, UpdateTxFirmware,
                (const char * seaFilename, const char * sedFilename),
                (override));
    MOCK_METHOD(float, GetUpdateTxFirmwareProgress, (), (override));
    MOCK_METHOD(const char *, GetUpdateTxFirmwareStatus, (), (override));
    MOCK_METHOD(void, SWTrigger, (), (override));
    MOCK_METHOD(void, SWNextTX, (), (override));
    MOCK_METHOD(void, EnableTestPatterns, (), (override));
    MOCK_METHOD(void, DisableTestPatterns, (), (override));
    MOCK_METHOD(void, SyncTestPatterns, (), (override));
    MOCK_METHOD(void, LockDMABuffer, (unsigned char * address, size_t length),
                (override));
    MOCK_METHOD(void, ReleaseDMABuffer, (unsigned char * address), (override));
};


class Us4OEMFactorySimpleParametersTest
        : public testing::TestWithParam<std::pair<TestUs4OEMSettings, ExpectedUs4RParameters>> {
};

TEST_P(Us4OEMFactorySimpleParametersTest, VerifyUs4OEMFactorySimpleParameters) {
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ExpectedUs4RParameters us4rParameters = GetParam().second;
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetPGAGain(us4rParameters.pgaGain));
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetLNAGain(us4rParameters.lnaGain));
    EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                SetDTGC(us4rParameters.dtgcAttEnabled,
                        us4rParameters.dtgcAttValue));
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetLPFCutoff(us4rParameters.lpfCutoff));
    EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                SetActiveTermination(us4rParameters.activeTerminationEnabled,
                                     us4rParameters.activeTerminationValue));
    Us4OEMFactoryImpl factory;

    factory.getUs4OEM(0, ius4oem, GetParam().first.getUs4OEMSettings());
}

INSTANTIATE_TEST_CASE_P

(Us4OEMFactorySetsSimpleIUs4OEMProperties, Us4OEMFactorySimpleParametersTest,
 testing::Values(
         std::pair{TestUs4OEMSettings{},
                   ExpectedUs4RParameters{}},
         std::pair{TestUs4OEMSettings{.pgaGain=24},
                   ExpectedUs4RParameters{.pgaGain=PGA_GAIN::PGA_GAIN_24dB}},
         std::pair{TestUs4OEMSettings{.lnaGain=12},
                   ExpectedUs4RParameters{.lnaGain=LNA_GAIN_GBL::LNA_GAIN_GBL_12dB}},
         std::pair{TestUs4OEMSettings{.lpfCutoff=(int) 15e6},
                   ExpectedUs4RParameters{.lpfCutoff=LPF_PROG::LPF_PROG_15MHz}}
 ));

class Us4OEMFactoryOptionalParametersTest
        : public testing::TestWithParam<std::pair<TestUs4OEMSettings, ExpectedUs4RParameters>> {
};

TEST_P(Us4OEMFactoryOptionalParametersTest,
       VerifyUs4OEMFactoryOptionalParameters) {
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ExpectedUs4RParameters us4rParameters = GetParam().second;
    if(GetParam().first.dtgcAttenuation.has_value()) {
        // NO disable
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetDTGC(EN_DIG_TGC::EN_DIG_TGC_DIS, _)).Times(0);
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetDTGC(EN_DIG_TGC::EN_DIG_TGC_EN,
                            us4rParameters.dtgcAttValue));
    } else {
        // NO enable
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetDTGC(EN_DIG_TGC::EN_DIG_TGC_EN, _)).Times(0);

        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetDTGC(EN_DIG_TGC::EN_DIG_TGC_DIS, _));
    }

    if(GetParam().first.activeTermination.has_value()) {
        // NO disable
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_DIS,
                                         us4rParameters.activeTerminationValue))
                .Times(0);
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_EN,
                                         us4rParameters.activeTerminationValue));
    } else {
        // NO enable
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_EN, _))
                .Times(0);
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    SetActiveTermination(ACTIVE_TERM_EN::ACTIVE_TERM_DIS, _));
    }
    Us4OEMFactoryImpl factory;

    factory.getUs4OEM(0, ius4oem, GetParam().first.getUs4OEMSettings());
}

INSTANTIATE_TEST_CASE_P

(Us4OEMFactorySetsOptionalIUs4OEMProperties,
 Us4OEMFactoryOptionalParametersTest,
 testing::Values(
         std::pair{TestUs4OEMSettings{.dtgcAttenuation=0},
                   ExpectedUs4RParameters{.dtgcAttValue=DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_0dB}},
         std::pair{TestUs4OEMSettings{.dtgcAttenuation={}},
                   ExpectedUs4RParameters{}}, // Any value is accepted
         std::pair{TestUs4OEMSettings{.activeTermination=200},
                   ExpectedUs4RParameters{.activeTerminationValue=GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_200}},
         std::pair{TestUs4OEMSettings{.activeTermination={}},
                   ExpectedUs4RParameters{}}
 ));


class Us4OEMFactoryTGCSamplesTest
        : public testing::TestWithParam<std::pair<TestUs4OEMSettings, ExpectedUs4RParameters>> {
};

TEST_P(Us4OEMFactoryTGCSamplesTest, VerifyUs4OEMFactorySimpleParameters) {
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    ExpectedUs4RParameters us4rParameters = GetParam().second;
    TGCCurve tgcCurve = GetParam().first.getUs4OEMSettings().getTGCSettings().getTGCSamples();

    if(tgcCurve.empty()) {
        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCDisable());
        // NO TGC enable
        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCEnable()).Times(0);
    } else {
        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCEnable());
        // NO TGC disable
        EXPECT_CALL(GET_MOCK_PTR(ius4oem), TGCDisable()).Times(0);
        EXPECT_CALL(GET_MOCK_PTR(ius4oem),
                    TGCSetSamples(
                            Pointwise(
                                    FloatEq(),
                                    us4rParameters.tgcSamplesNormalized), _));
    }

    Us4OEMFactoryImpl factory;

    factory.getUs4OEM(0, ius4oem, GetParam().first.getUs4OEMSettings());
}

INSTANTIATE_TEST_CASE_P

(Us4OEMFactorySetsTGCSettings,
 Us4OEMFactoryTGCSamplesTest,
 testing::Values(
         // NO TGC
         std::pair{TestUs4OEMSettings{.tgcSamples={}},
                   ExpectedUs4RParameters{}},
         // TGC samples set
         std::pair{TestUs4OEMSettings{
                 .pgaGain=30,
                 .lnaGain=24,
                 .tgcSamples={30, 35, 40}},
                   ExpectedUs4RParameters{.tgcSamplesNormalized={0.4, 0.525,
                                                                 0.65}}}
 ));

// Mappings.

TEST(Us4OEMFactoryTest, WorksForConsistentMapping) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();

    // Mapping includes groups of 32 channel, each has the same permutation
    std::vector<uint8_t> channelMapping = getRange<uint8_t>(0, 128, 1);

    for(int i = 0; i < 4; ++i) {
        std::swap(channelMapping[i * 32], channelMapping[(i + 1) * 32 - 1]);
    }

    TestUs4OEMSettings cfg{.channelMapping=std::vector<ChannelIdx>(
            std::begin(channelMapping), std::end(channelMapping))};
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
    factory.getUs4OEM(0, ius4oem, cfg.getUs4OEMSettings());
}

TEST(Us4OEMFactoryTest, WorksForInconsistentMapping) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();

    // Mapping includes groups of 32 channel, each has the same permutation
    std::vector<uint8_t> channelMapping = getRange<uint8_t>(0, 128, 1);

    for(int i = 0; i < 2; ++i) {
        std::swap(channelMapping[i * 32], channelMapping[(i + 1) * 32 - 1]);
    }

    for(int i = 2; i < 4; ++i) {
        std::swap(channelMapping[i * 32 + 1], channelMapping[(i + 1) * 32 - 2]);
    }

    TestUs4OEMSettings cfg{.channelMapping=std::vector<ChannelIdx>(
            std::begin(channelMapping), std::end(channelMapping))};
    Us4OEMFactoryImpl factory;
    // Expect
    EXPECT_CALL(GET_MOCK_PTR(ius4oem), SetRxChannelMapping(_, _)).Times(0);
    // Run
    factory.getUs4OEM(0, ius4oem, cfg.getUs4OEMSettings());
}

// Tx channel mapping
TEST(Us4OEMFactoryTest, WorksForTxChannelMapping) {
    // Given
    std::unique_ptr<IUs4OEM> ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
    std::vector<ChannelIdx> channelMapping = getRange<ChannelIdx>(0, 128, 1);
    TestUs4OEMSettings cfg{.channelMapping=channelMapping};
    Us4OEMFactoryImpl factory;
    // Expect
    {
        InSequence seq;
        for(ChannelIdx i = 0; i < Us4OEMImpl::N_TX_CHANNELS; ++i) {
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
    factory.getUs4OEM(0, ius4oem, cfg.getUs4OEMSettings());
}

}
