#include <arrus/core/devices/us4r/FrameChannelMappingImpl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ProbeAdapterImpl.h"

#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImplBase.h"

namespace {

using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;
using ::arrus::framework::NdArray;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ByMove;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Property;
using ::testing::Return;

const ChannelIdx DEFAULT_NCHANNELS = 64;

struct TestTxRxParams {

    TestTxRxParams() {
        for (int i = 0; i < 32; ++i) {
            rxAperture[i] = true;
        }
    }

    BitMask txAperture = getNTimes(true, DEFAULT_NCHANNELS);
    std::vector<float> txDelays = getNTimes(0.0f, DEFAULT_NCHANNELS);
    ops::us4r::Pulse pulse{2.0e6f, 2.5f, true};
    BitMask rxAperture = getNTimes(false, DEFAULT_NCHANNELS);
    uint32 decimationFactor = 1;
    float pri = 100e-6f;
    Interval<uint32> sampleRange{0, 4096};
    Tuple<ChannelIdx> rxPadding{0, 0};

    [[nodiscard]] TxRxParameters getTxRxParameters() const {
        return TxRxParameters(txAperture, txDelays, pulse, rxAperture, sampleRange, decimationFactor, pri, rxPadding);
    }
};

BitMask getDefaultTxAperture(ChannelIdx nchannels) { return BitMask(nchannels, true); }

BitMask getDefaultRxAperture(ChannelIdx nchannels) {
    BitMask aperture(nchannels, false);
    ::arrus::setValuesInRange(aperture, 0, Us4OEMImpl::N_RX_CHANNELS, true);
    return aperture;
}

std::vector<float> getDefaultTxDelays(ChannelIdx nchannels) { return getNTimes(0.0f, nchannels); }

Us4OEMBuffer createUs4OEMBuffer(FrameChannelMapping::FrameNumber nFrames, ChannelIdx nChannels, uint32_t nSamples) {
    std::vector<Us4OEMBufferElementPart> parts;
    size_t partAddress = 0;
    auto dataType = NdArray::DataType::INT16;
    NdArray::Shape shape{nFrames * nSamples, nChannels};
    const size_t partSize = nSamples * nChannels * NdArray::getDataTypeSize(dataType);
    for (uint16 frame = 0; frame < nFrames; ++frame) {
        parts.push_back(Us4OEMBufferElementPart{partAddress, partSize, frame, nSamples});
        partAddress += partSize;
    }
    size_t totalSize = partAddress;
    // Single element buffer
    Us4OEMBuffer buffer({Us4OEMBufferElement(0, totalSize, 0, shape, dataType)}, parts);
    return buffer;
}

std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> createEmptySetTxRxResult(FrameChannelMapping::Us4OEMNumber us4oem,
                                                                               FrameChannelMapping::FrameNumber nFrames,
                                                                               ChannelIdx nChannels,
                                                                               uint32_t nSamples = 4096) {
    FrameChannelMappingBuilder builder(nFrames, nChannels);
    for (int i = 0; i < nFrames; ++i) {
        for (int j = 0; j < nChannels; ++j) {
            builder.setChannelMapping(i, j, us4oem, i, j);
        }
    }
    return std::make_tuple(createUs4OEMBuffer(nFrames, nChannels, nSamples), builder.build());
}

class MockUs4OEM : public Us4OEMImplBase {
public:
    explicit MockUs4OEM(Ordinal id) : Us4OEMImplBase(DeviceId(DeviceType::Us4OEM, id)) {}

    MOCK_METHOD((std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>), setTxRxSequence,
                (const TxRxParamsSequence &seq, const ::arrus::ops::us4r::TGCCurve &tgc, uint16 rxBufferSize,
                 uint16 batchSize, std::optional<float> sri, arrus::ops::us4r::Scheme::WorkMode workMode,
                 const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc,
                 const std::vector<arrus::framework::NdArray> &txDelayProfiles),
                (override));
    MOCK_METHOD(Interval<Voltage>, getAcceptedVoltageRange, (), (override));
    MOCK_METHOD(float, getSamplingFrequency, (), (override));
    MOCK_METHOD(float, getCurrentSamplingFrequency, (), (const, override));
    MOCK_METHOD(void, startTrigger, (), (override));
    MOCK_METHOD(void, stopTrigger, (), (override));
    MOCK_METHOD(void, start, (), (override));
    MOCK_METHOD(void, stop, (), (override));
    MOCK_METHOD(void, syncTrigger, (), (override));
    MOCK_METHOD(bool, isMaster, (), (override));
    MOCK_METHOD(void, setRxSettings, (const RxSettings &cfg), (override));
    MOCK_METHOD(Ius4OEMRawHandle, getIUs4oem, (), (override));
    MOCK_METHOD(void, enableSequencer, (uint16 startEntry), (override));
    MOCK_METHOD(std::vector<uint8_t>, getChannelMapping, (), (override));
    MOCK_METHOD(float, getFPGATemperature, (), (override));
    MOCK_METHOD(void, setTestPattern, (Us4OEMImpl::RxTestPattern), (override));
    MOCK_METHOD(void, checkFirmwareVersion, (), (override));
    MOCK_METHOD(void, checkState, (), (override));
    MOCK_METHOD(uint32, getFirmwareVersion, (), (override));
    MOCK_METHOD(uint32, getTxFirmwareVersion, (), (override));
    MOCK_METHOD(void, setAfeDemod,
                (float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                 size_t nCoefficients),
                (override));
    MOCK_METHOD(void, disableAfeDemod, (), (override));
    MOCK_METHOD(uint16_t, getAfe, (uint8_t address), (override));
    MOCK_METHOD(void, setAfe, (uint8_t address, uint16_t value), (override));
    MOCK_METHOD(float, getUCDMeasuredVoltage, (uint8_t), (override));
    MOCK_METHOD(float, getMeasuredHVPVoltage, (), (override));
    MOCK_METHOD(float, getMeasuredHVMVoltage, (), (override));
    MOCK_METHOD(float, getFPGAWallclock, (), (override));
    MOCK_METHOD(void, setHpfCornerFrequency, (uint32_t), (override));
    MOCK_METHOD(void, disableHpf, (), (override));
    MOCK_METHOD(float, getUCDTemperature, (), (override));
    MOCK_METHOD(float, getUCDExternalTemperature, (), (override));
    MOCK_METHOD(const char *, getSerialNumber, (), (override));
    MOCK_METHOD(const char *, getRevision, (), (override));

    MOCK_METHOD(uint32_t, getTxOffset, (), (override));
    MOCK_METHOD(uint32_t, getOemVersion, (), (override));
    MOCK_METHOD(void, setSubsequence, (uint16 start, uint16 end, bool syncMode, const std::optional<float> &sri), (override));
    MOCK_METHOD(void, clearCallbacks, (), (override));
    MOCK_METHOD(HVPSMeasurement, getHVPSMeasurement, (), (override));
    MOCK_METHOD(float, setHVPSSyncMeasurement, (uint16_t nSamples, float frequency), (override));
    MOCK_METHOD(void, setMaximumPulseLength, (std::optional<float> maxPulseLength), (override));
    MOCK_METHOD(void, waitForHVPSMeasurementDone, (std::optional<long long> timeout), (override));
    MOCK_METHOD(void, setWaitForHVPSMeasurementDone, (), (override));
    MOCK_METHOD(void, sync, (std::optional<long long> timeout), (override));
};

class AbstractProbeAdapterImplTest : public ::testing::Test {
    using NiceMockHandle = std::unique_ptr<::testing::NiceMock<MockUs4OEM>>;

protected:
    void SetUp() override {
        us4oems.push_back(std::make_unique<::testing::NiceMock<MockUs4OEM>>(0));
        us4oemsPtr.push_back(us4oems[0].get());
        us4oems.push_back(std::make_unique<::testing::NiceMock<MockUs4OEM>>(1));
        us4oemsPtr.push_back(us4oems[1].get());

        ON_CALL(*us4oems[0], getChannelMapping()).WillByDefault(Return(defaultChannelMapping[0]));
        ON_CALL(*us4oems[1], getChannelMapping()).WillByDefault(Return(defaultChannelMapping[1]));

        probeAdapter = std::make_unique<ProbeAdapterImpl>(
            DeviceId(DeviceType::ProbeAdapter, 0), ProbeAdapterModelId("test", "test"), us4oemsPtr, getNChannels(),
            getChannelMapping(), ::arrus::devices::us4r::IOSettings());
    }

    virtual ProbeAdapterImpl::ChannelMapping getChannelMapping() = 0;

    virtual ChannelIdx getNChannels() { return DEFAULT_NCHANNELS; }

    std::vector<NiceMockHandle> us4oems;
    std::vector<Us4OEMImplBase::RawHandle> us4oemsPtr;
    ProbeAdapterImpl::Handle probeAdapter;
    TGCCurve defaultTGCCurve;
    std::vector<std::vector<uint8_t>> defaultChannelMapping = {getRange<uint8_t>(0, 128), getRange<uint8_t>(0, 128)};
};

class ProbeAdapter64ChannelsTest : public AbstractProbeAdapterImplTest {
    // An adapter with 64 channels.
    // 0-32 channels to us4oem:0
    // 32-64 channels to us4oem:1
    ProbeAdapterImpl::ChannelMapping getChannelMapping() override {
        ProbeAdapterImpl::ChannelMapping mapping(getNChannels());
        for (ChannelIdx ch = 0; ch < getNChannels(); ++ch) {
            mapping[ch] = {ch / 2, ch % 32};
        }
        return mapping;
    }
};

// ------------------------------------------ Test validation
TEST_F(ProbeAdapter64ChannelsTest, ChecksRxApertureSize) {
    BitMask rxAperture(128, false);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.rxAperture = rxAperture)).getTxRxParameters()};

    // Throw: nchannels = 64, rx aperture size = 128
    EXPECT_THROW(probeAdapter->setTxRxSequence(seq, defaultTGCCurve), IllegalArgumentException);
}

TEST_F(ProbeAdapter64ChannelsTest, ChecksTxApertureSize) {
    BitMask txAperture(32, false);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.txAperture = txAperture)).getTxRxParameters()};

    // Throw: nchannels = 64, aperture size = 32
    EXPECT_THROW(probeAdapter->setTxRxSequence(seq, defaultTGCCurve), IllegalArgumentException);
}

TEST_F(ProbeAdapter64ChannelsTest, ChecksTxDelaysSize) {
    std::vector<float> txDelays(65, false);

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.txDelays = txDelays)).getTxRxParameters()};

    // Throw: nchannels = 64, number of delays = 65
    EXPECT_THROW(probeAdapter->setTxRxSequence(seq, defaultTGCCurve), IllegalArgumentException);
}

// ------------------------------------------ Test aperture/delays distribution

class ProbeAdapterChannelMapping1Test : public AbstractProbeAdapterImplTest {
    // An adapter with 64 channels.
    // 0-32 channels to us4oem:0
    // 32-64 channels to us4oem:1
    ProbeAdapterImpl::ChannelMapping getChannelMapping() override {
        ProbeAdapterImpl::ChannelMapping mapping(getNChannels());
        for (ChannelIdx ch = 0; ch < getNChannels(); ++ch) {
            mapping[ch] = {ch / 32, ch % 32};
        }
        return mapping;
    }
};

#define EXPECT_SEQUENCE_PROPERTY_NFRAMES(deviceId, matcher, nFrames)                                                   \
    do {                                                                                                               \
                                                                                                                       \
        EXPECT_CALL(*(us4oems[deviceId].get()), setTxRxSequence(matcher, _, _, _, _, _, _, _))                         \
            .WillOnce(Return(ByMove(createEmptySetTxRxResult(deviceId, nFrames, 32))));                                \
    } while (0)

#define EXPECT_SEQUENCE_PROPERTY(deviceId, matcher) EXPECT_SEQUENCE_PROPERTY_NFRAMES(deviceId, matcher, 1)

#define SET_TX_RX_SEQUENCE(probeAdapter, seq) probeAdapter->setTxRxSequence(seq, defaultTGCCurve)

#define US4OEM_MOCK_SET_TX_RX_SEQUENCE() setTxRxSequence(_, _, _, _, _, _, _, _)

TEST_F(ProbeAdapterChannelMapping1Test, DistributesTxAperturesCorrectly) {
    BitMask txAperture(64, false);
    ::arrus::setValuesInRange(txAperture, 20, 40, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.txAperture = txAperture)).getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 20, 32, true);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 8, true);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMapping1Test, DistributesRxAperturesCorrectly) {
    BitMask rxAperture(64, false);
    ::arrus::setValuesInRange(rxAperture, 15, 51, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.rxAperture = rxAperture)).getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 15, 32, true);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getRxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 19, true);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getRxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMapping1Test, DistributesTxDelaysCorrectly) {
    std::vector<float> delays(64, 0.0f);
    for (int i = 18; i < 44; ++i) {
        delays[i] = i * 5e-6;
    }
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.txDelays = delays)).getTxRxParameters()};

    std::vector<float> delays0(Us4OEMImpl::N_TX_CHANNELS, 0);
    for (int i = 18; i < 32; ++i) {
        delays0[i] = i * 5e-6;
    }
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxDelays, delays0)));

    std::vector<float> delays1(Us4OEMImpl::N_TX_CHANNELS, 0);
    for (int i = 0; i < 44 - 32; ++i) {
        delays1[i] = (i + 32) * 5e-6;
    }
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxDelays, delays1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMapping1Test, DistributesTxAperturesCorrectlySingleUs4OEM0) {
    BitMask txAperture(64, false);
    ::arrus::setValuesInRange(txAperture, 10, 21, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.txAperture = txAperture)).getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 10, 21, true);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_ADDR_CHANNELS, false);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMapping1Test, DistributesTxAperturesCorrectlySingleUs4OEM1) {
    BitMask txAperture(64, false);
    ::arrus::setValuesInRange(txAperture, 42, 61, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams, (x.txAperture = txAperture)).getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_ADDR_CHANNELS, false);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 10, 29, true);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

class ProbeAdapterChannelMappingEsaote3Test : public AbstractProbeAdapterImplTest {
    // An adapter with 192 channels.
    // 0-32, 64-96, 128-160 channels to us4oem:0
    // 32-64, 96-128, 160-192 channels to us4oem:1
protected:
    ProbeAdapterImpl::ChannelMapping getChannelMapping() override {
        ProbeAdapterImpl::ChannelMapping mapping(getNChannels());
        for (ChannelIdx ch = 0; ch < getNChannels(); ++ch) {
            auto group = ch / 32;
            auto module = group % 2;
            mapping[ch] = {module, ch % 32 + 32 * (group / 2)};
        }
        return mapping;
    }

    ChannelIdx getNChannels() override { return 192; }
};

TEST_F(ProbeAdapterChannelMappingEsaote3Test, DistributesTxAperturesCorrectlySingleUs4OEM) {
    BitMask txAperture(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture, 65, 80, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = txAperture, x.rxAperture = getDefaultRxAperture(getNChannels()),
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 33, 48, true);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, DistributesTxAperturesCorrectlyTwoSubapertures) {
    BitMask txAperture(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture, 128 + 14, 128 + 40, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = txAperture, x.rxAperture = getDefaultRxAperture(getNChannels()),
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 64 + 14, 64 + 32, true);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 64 + 0, 64 + 8, true);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, DistributesTxAperturesCorrectlyThreeSubapertures) {
    BitMask txAperture(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture, 16, 80, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = txAperture, x.rxAperture = getDefaultRxAperture(getNChannels()),
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 16, 48, true);
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 32, true);
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, DistributesTxAperturesWithGapsCorrectly) {
    BitMask txAperture(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture, 0 + 8, 160 + 30, true);

    txAperture[0 + 14] = txAperture[0 + 17] = txAperture[32 + 23] = txAperture[32 + 24] = txAperture[64 + 25] =
        txAperture[160 + 7] = false;
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = txAperture, x.rxAperture = getDefaultRxAperture(getNChannels()),
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 8, 96, true);
    expectedTxAp0[0 + 14] = expectedTxAp0[0 + 17] = expectedTxAp0[32 + 25] = false;
    EXPECT_SEQUENCE_PROPERTY(0, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp0)));

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 64 + 30, true);
    expectedTxAp1[0 + 23] = expectedTxAp1[0 + 24] = expectedTxAp1[64 + 7] = false;
    EXPECT_SEQUENCE_PROPERTY(1, ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp1)));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, DistributesAperturesCorrectlyForMultipleRxApertures) {
    BitMask txAperture(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture, 0 + 8, 160 + 30, true);

    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 16, 96, true);
    rxAperture[0 + 18] = rxAperture[32 + 23] = false;
    // There should be two apertures: [16, 80], [80, 100] with two gaps: 18, 55

    txAperture[0 + 14] = txAperture[0 + 17] = txAperture[32 + 23] = txAperture[32 + 24] = txAperture[64 + 25] =
        txAperture[160 + 7] = false;
    std::vector<TxRxParameters> seq = {ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
                                                               x.txDelays = getDefaultTxDelays(getNChannels())))
                                           .getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 8, 96, true);
    expectedTxAp0[0 + 14] = expectedTxAp0[0 + 17] = expectedTxAp0[32 + 25] = false;

    BitMask expectedRxAp00(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedRxAp00, 16, 32 + 80 - 64, true);
    expectedRxAp00[18] = false;
    // TODO(pjarosik) this should be done in a pretty more clever way, to minimize
    // potential transfers that are needed
    // Instead, the next one channel can be used here
    expectedRxAp00[18 + 32] = true;
    BitMask expectedRxAp01(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedRxAp01, 32 + 80 - 64, 64, true);
    // 18+32 is already covered by op 0
    expectedRxAp01[18 + 32] = false;

    EXPECT_SEQUENCE_PROPERTY_NFRAMES(0,
                                     // Tx aperture should stay the same.
                                     // Rx aperture should be adjusted appropriately.
                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp0),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp00)),
                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp0),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp01))),
                                     2);

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 64 + 30, true);
    expectedTxAp1[0 + 23] = expectedTxAp1[0 + 24] = expectedTxAp1[64 + 7] = false;

    BitMask expectedRxAp10(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedRxAp10, 0, 32, true);
    expectedRxAp10[23] = false;

    BitMask expectedRxAp11(Us4OEMImpl::N_ADDR_CHANNELS, false);
    // second aperture should be empty
    EXPECT_SEQUENCE_PROPERTY_NFRAMES(1,
                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp10)),
                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp11))),
                                     2);

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test,
       DistributesAperturesCorrectlyForMultipleRxAperturesForFrameMetadataUs4OEM) {
    // It should keep tx aperture on the second module even if there is no rx aperture for this module
    BitMask txAperture(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture, 0 + 9, 160 + 31, true);

    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 16, 32, true);
    ::arrus::setValuesInRange(rxAperture, 64 + 16, 64 + 32, true);
    rxAperture[0 + 18] = rxAperture[64 + 23] = false;

    txAperture[0 + 14] = txAperture[0 + 17] = txAperture[32 + 23] = txAperture[32 + 24] = txAperture[64 + 25] =
        txAperture[160 + 7] = false;
    std::vector<TxRxParameters> seq = {ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
                                                               x.txDelays = getDefaultTxDelays(getNChannels())))
                                           .getTxRxParameters()};
    BitMask expectedTxAp0(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp0, 9, 96, true);
    expectedTxAp0[0 + 14] = expectedTxAp0[0 + 17] = expectedTxAp0[32 + 25] = false;

    BitMask expectedRxAp00(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedRxAp00, 16, 32, true);
    expectedRxAp00[18] = false;
    expectedRxAp00[32 + 18] = true;
    BitMask expectedRxAp01(Us4OEMImpl::N_ADDR_CHANNELS, false);
    ::arrus::setValuesInRange(expectedRxAp01, 32 + 16, 32 + 32, true);
    expectedRxAp01[32 + 23] = false;
    expectedRxAp01[32 + 18] = false;

    EXPECT_SEQUENCE_PROPERTY_NFRAMES(0,
                                     // Tx aperture should stay the same.
                                     // Rx aperture should be adjusted appropriately.
                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp0),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp00)),
                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp0),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp01))),
                                     2);

    BitMask expectedTxAp1(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 64 + 31, true);
    expectedTxAp1[0 + 23] = expectedTxAp1[0 + 24] = expectedTxAp1[64 + 7] = false;

    // rx apertures should be empty
    BitMask expectedRxAp10(Us4OEMImpl::N_ADDR_CHANNELS, false);
    BitMask expectedRxAp11(Us4OEMImpl::N_ADDR_CHANNELS, false);

    EXPECT_SEQUENCE_PROPERTY_NFRAMES(1,
                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp10)),
                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp11))),
                                     2);

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, DistributesTxAperturesTwoOperations) {
    BitMask txAperture0(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture0, 20, 64 + 20, true);
    BitMask txAperture1(getNChannels(), false);
    ::arrus::setValuesInRange(txAperture1, 23, 64 + 23, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = txAperture0, x.rxAperture = getDefaultRxAperture(getNChannels()),
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters(),
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = txAperture1, x.rxAperture = getDefaultRxAperture(getNChannels()),
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    BitMask expectedTxAp00(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp00, 20, 32 + 20, true);
    BitMask expectedTxAp01(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp01, 23, 32 + 23, true);
    EXPECT_SEQUENCE_PROPERTY_NFRAMES(0,
                                     ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp00),
                                                 Property(&TxRxParameters::getTxAperture, expectedTxAp01)),
                                     2);

    BitMask expectedTxAp10(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp10, 0, 32, true);
    BitMask expectedTxAp11(Us4OEMImpl::N_TX_CHANNELS, false);
    ::arrus::setValuesInRange(expectedTxAp11, 0, 32, true);
    EXPECT_SEQUENCE_PROPERTY_NFRAMES(1,
                                     ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp10),
                                                 Property(&TxRxParameters::getTxAperture, expectedTxAp11)),
                                     2);

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

// ------------------------------------------ Test Frame Channel Mapping
TEST_F(ProbeAdapterChannelMappingEsaote3Test, ProducesCorrectFCMSingleDistributedOperation) {
    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 16, 72, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture,
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0; i < 32; ++i) {
        if (i < 24) {
            builder0.setChannelMapping(0, i, 0, 0, i);
        } else {
            builder0.setChannelMapping(0, i, 0, 0, -1);
        }
    }
    auto fcm0 = builder0.build();

    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0; i < 32; ++i) {
        builder1.setChannelMapping(0, i, 1, 0, i);
    }
    auto fcm1 = builder1.build();
    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));

    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);

    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
    EXPECT_EQ(72 - 16, fcm->getNumberOfLogicalChannels());

    for (int i = 0; i < 16; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(0, address.getUs4oem());
        EXPECT_EQ(0, address.getFrame());
        EXPECT_EQ(i, address.getChannel());
    }

    for (int i = 16; i < 16 + 32; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(1, address.getUs4oem());
        EXPECT_EQ(0, address.getFrame());
        EXPECT_EQ(address.getChannel(), i - 16);
    }

    for (int i = 16 + 32; i < 56; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(0, address.getUs4oem());
        EXPECT_EQ(0, address.getFrame());
        EXPECT_EQ(address.getChannel(), i - 32);
    }

    // Make sure the correct frame offsets are set.
    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, ProducesCorrectFCMSingleDistributedOperationWithGaps) {
    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 16, 73, true);
    // Channels 20, 30 and 40 were masked for given us4oem and data is missing.
    // Still, the input rx aperture stays as is.

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture,
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0, j = -1; i < 32; ++i) {
        int currentJ = -1;
        // channels were marked by the us4oem that are missing
        if (i != 20 - 16 && i != 30 - 16 && i <= 25) {
            currentJ = ++j;
        }
        builder0.setChannelMapping(0, i, 0, 0, currentJ);
    }
    auto fcm0 = builder0.build();

    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0, j = -1; i < 32; ++i) {
        int currentJ = -1;
        if (i != 40 - 32) {
            currentJ = ++j;
        }
        builder1.setChannelMapping(0, i, 1, 0, currentJ);
    }
    auto fcm1 = builder1.build();

    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));

    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);

    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
    EXPECT_EQ(73 - 16, fcm->getNumberOfLogicalChannels());

    std::vector<FrameChannelMapping::FrameNumber> expectedFrames;
    std::vector<FrameChannelMapping::Us4OEMNumber> expectedUs4oems;
    for (int i = 16; i < 32; ++i) {
        expectedUs4oems.push_back(0);
        expectedFrames.push_back(0);
    }
    for (int i = 32; i < 64; ++i) {
        expectedUs4oems.push_back(1);
        expectedFrames.push_back(0);
    }
    for (int i = 64; i < 73; ++i) {
        expectedUs4oems.push_back(0);
        expectedFrames.push_back(0);
    }
    std::vector<int8> expectedChannels = {0,  1,  2,  3,  -1, 4,  5,  6,  7,  8,  9,  10, 11, 12, -1, 13, 0,  1,  2,
                                          3,  4,  5,  6,  7,  -1, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                          21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 14, 15, 16, 17, 18, 19, 20, 21, 22};

    for (int i = 0; i < 73 - 16; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(expectedUs4oems[i], address.getUs4oem());
        EXPECT_EQ(expectedFrames[i], address.getFrame());
        EXPECT_EQ(expectedChannels[i], address.getChannel());
    }
    // Make sure the correct frame offsets are set.
    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, ProducesCorrectFCMForMultiOpRxAperture) {
    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 48, 128, true);
    // RxNOP - the second operation on us4oem
    // Ops: us4oem0: 32-64 (64-96), Rx NOP, us4oem1: 16-48, 48-64
    // Channel 99 (us4oem:1 channel 32+3) is masked and data is missing.
    // Still, the input rx aperture stays as is.

    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture,
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};
    // FCM from the us4oem:0
    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
    // The second op is Rx NOP.
    for (int i = 0; i < 32; ++i) {
        builder0.setChannelMapping(0, i, 0, 0, i);
    }
    auto fcm0 = builder0.build();

    FrameChannelMappingBuilder builder1(2, Us4OEMImpl::N_RX_CHANNELS);
    // First frame:
    for (int i = 0, j = -1; i < 32; ++i) {
        int currentJ = -1;
        if (i != 16 + 3) {
            currentJ = ++j;
            builder1.setChannelMapping(0, i, 1, 0, currentJ);
        } else {
            builder1.setChannelMapping(0, i, 1, 0, FrameChannelMapping::UNAVAILABLE);
        }
    }
    // Second frame:
    for (int i = 0; i < 32; ++i) {
        if (i < 16) {
            builder1.setChannelMapping(1, i, 1, 1, i);
        } else {
            builder1.setChannelMapping(1, i, 1, 1, FrameChannelMapping::UNAVAILABLE);
        }
    }
    auto fcm1 = builder1.build();

    auto us4oemBuffer = createUs4OEMBuffer(2, 32, 4096);

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));

    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);

    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
    EXPECT_EQ(128 - 48, fcm->getNumberOfLogicalChannels());

    std::vector<FrameChannelMapping::Us4OEMNumber> expectedUs4oems;
    std::vector<FrameChannelMapping::FrameNumber> expectedFrames;
    std::vector<int8> expectedChannels;

    // Us4OEM:1, frame 0, channels 0-16
    for (int i = 48; i < 64; ++i) {
        expectedUs4oems.push_back(1);
        expectedFrames.push_back(0);
        expectedChannels.push_back(i - 48);
    }
    // Us4OEM:0
    for (int i = 64; i < 96; ++i) {
        expectedUs4oems.push_back(0);
        expectedFrames.push_back(0);
        expectedChannels.push_back(i - 64);
    }
    // Us4OEM:1, frame 0, channels 16-32
    for (int i = 96; i < 96 + 15; ++i) {// 15 because there will be one -1
        expectedUs4oems.push_back(1);
        expectedFrames.push_back(0);
        if (i == 99 && expectedChannels[expectedChannels.size() - 1] != FrameChannelMapping::UNAVAILABLE) {
            expectedChannels.push_back(FrameChannelMapping::UNAVAILABLE);
            --i;
        } else {
            expectedChannels.push_back(i - 96 + 16);
        }
    }
    // Us4OEM:1, frame 1
    for (int i = 96 + 16; i < 128; ++i) {
        expectedUs4oems.push_back(1);
        expectedFrames.push_back(1);
        expectedChannels.push_back(i - (96 + 16));
    }

    // VALIDATE
    for (int i = 0; i < 128 - 48; ++i) {
        auto address = fcm->getLogical(0, i);
        EXPECT_EQ(expectedUs4oems[i], address.getUs4oem());
        EXPECT_EQ(expectedFrames[i], address.getFrame());
        EXPECT_EQ(expectedChannels[i], address.getChannel());
    }
    // Make sure the correct frame offsets are set.
    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
}

// Currently padding impacts the output frame channel mapping
TEST_F(ProbeAdapterChannelMappingEsaote3Test, AppliesPaddingToFCMCorrectly) {
    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 0, 16, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture,
                                x.txDelays = getDefaultTxDelays(getNChannels()), x.rxPadding = {16, 0}))
            .getTxRxParameters()};
    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0; i < 32; ++i) {
        if (i < 16) {
            builder0.setChannelMapping(0, i, 0, 0, i);
        } else {
            builder0.setChannelMapping(0, i, 0, 0, -1);
        }
    }
    auto fcm0 = builder0.build();

    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
    // No active channels
    auto fcm1 = builder1.build();

    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));

    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);

    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
    EXPECT_EQ(32, fcm->getNumberOfLogicalChannels());// 16 active + 16 rx padding

    for (int i = 0; i < 16; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(0, address.getFrame());
        ASSERT_EQ(address.getChannel(), FrameChannelMapping::UNAVAILABLE);
    }

    for (int i = 16; i < 32; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(0, address.getUs4oem());
        ASSERT_EQ(0, address.getFrame());
        ASSERT_EQ(address.getChannel(), i - 16);
    }
    // Make sure the correct frame offsets are set.
    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
}

// The same as above, but with aperture using two modules
TEST_F(ProbeAdapterChannelMappingEsaote3Test, AppliesPaddingToFCMCorrectlyRxApertureUsingTwoModules) {
    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 0, 49, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture,
                                x.txDelays = getDefaultTxDelays(getNChannels()), x.rxPadding = {15, 0}))
            .getTxRxParameters()};
    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0; i < 32; ++i) {
        builder0.setChannelMapping(0, i, 0, 0, i);
    }
    auto fcm0 = builder0.build();

    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0; i < 32; ++i) {
        if (i < 17) {
            builder1.setChannelMapping(0, i, 1, 0, i);
        } else {
            builder1.setChannelMapping(0, i, 1, 0, FrameChannelMapping::UNAVAILABLE);
        }
    }
    auto fcm1 = builder1.build();

    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));

    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);

    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
    EXPECT_EQ(64, fcm->getNumberOfLogicalChannels());// 49 active + 15 rx padding

    for (int i = 0; i < 15; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(address.getChannel(), FrameChannelMapping::UNAVAILABLE);
    }
    for (int i = 15; i < 15 + 32; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(0, address.getUs4oem());
        ASSERT_EQ(0, address.getFrame());
        ASSERT_EQ(address.getChannel(), i - 15);
    }
    for (int i = 15 + 32; i < 64; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(1, address.getUs4oem());
        ASSERT_EQ(0, address.getFrame());
        ASSERT_EQ(address.getChannel(), i - (15 + 32));
    }
    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, AppliesPaddingToFCMCorrectlyRightSide) {
    BitMask rxAperture(getNChannels(), false);
    ::arrus::setValuesInRange(rxAperture, 176, 192, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture,
                                x.txDelays = getDefaultTxDelays(getNChannels()), x.rxPadding = {0, 16}))
            .getTxRxParameters()};
    FrameChannelMappingBuilder builder0(0, Us4OEMImpl::N_RX_CHANNELS);
    // No output
    auto fcm0 = builder0.build();

    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
    for (int i = 0; i < 32; ++i) {
        if (i < 16) {
            builder1.setChannelMapping(0, i, 1, 0, i);
        } else {
            builder1.setChannelMapping(0, i, 1, 0, FrameChannelMapping::UNAVAILABLE);
        }
    }
    auto fcm1 = builder1.build();

    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));

    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));

    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);

    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
    EXPECT_EQ(32, fcm->getNumberOfLogicalChannels());// 16 active + 16 rx padding

    for (int i = 0; i < 16; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(1, address.getUs4oem());
        ASSERT_EQ(0, address.getFrame());
        ASSERT_EQ(address.getChannel(), i);
    }
    for (int i = 16; i < 32; ++i) {
        auto address = fcm->getLogical(0, i);
        ASSERT_EQ(address.getChannel(), FrameChannelMapping::UNAVAILABLE);
    }
    EXPECT_EQ(0, fcm->getFirstFrame(1));// Us4OEM:1
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, CalculatesCorrectRxDelay) {
    std::vector<float> delays0(getNChannels(), 0.0f);
    std::vector<float> delays1(getNChannels(), 0.0f);
    BitMask txAperture(getNChannels(), true);
    BitMask rxAperture(getNChannels(), true);
    // Partially filled
    BitMask txAperture1(getNChannels(), false);
    std::fill(std::begin(txAperture1), std::begin(txAperture1) + 10, true);

    ops::us4r::Pulse pulse0{2.0e6f, 2.0f, false};
    ops::us4r::Pulse pulse1{3.0e6f, 3.0f, true};
    for (int i = 0; i < getNChannels(); ++i) {
        delays0[i] = i * 10e-7;
    }
    for (int i = 0; i < getNChannels(); ++i) {
        delays1[i] = 0.0f;
    }
    std::vector<TxRxParameters> seq = {// Linearly increasing TX delays.
                                       ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
                                                               x.txDelays = delays0, x.pulse = pulse0))
                                           .getTxRxParameters(),
                                       // All TX delays the same.
                                       ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
                                                               x.txDelays = delays1, x.pulse = pulse1))
                                           .getTxRxParameters(),
                                       // Partial TX aperture.
                                       ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture1, x.rxAperture = rxAperture,
                                                               x.txDelays = delays0, x.pulse = pulse1))
                                           .getTxRxParameters()};

    float rxDelay0 = *std::max_element(std::begin(delays0), std::end(delays0))
        + 1.0f / pulse0.getCenterFrequency() * pulse0.getNPeriods();
    float rxDelay1 = *std::max_element(std::begin(delays1), std::end(delays1))
        + 1.0f / pulse1.getCenterFrequency() * pulse1.getNPeriods();
    float rxDelay2 = *std::max_element(std::begin(delays0), std::begin(delays0) + 10)
        + 1.0f / pulse1.getCenterFrequency() * pulse1.getNPeriods();

    EXPECT_SEQUENCE_PROPERTY_NFRAMES(
        0,
        ElementsAre(Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay0),
                    Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay1), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay2), Property(&TxRxParameters::getRxDelay, rxDelay2),
                    Property(&TxRxParameters::getRxDelay, rxDelay2)),
        9);
    EXPECT_SEQUENCE_PROPERTY_NFRAMES(
        1,
        ElementsAre(Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay0),
                    Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay1), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay2), Property(&TxRxParameters::getRxDelay, rxDelay2),
                    Property(&TxRxParameters::getRxDelay, rxDelay2)),
        9);
    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

TEST_F(ProbeAdapterChannelMappingEsaote3Test, SetsSubapertureCorrectly) {
    BitMask rxAperture64(getNChannels(), false);
    BitMask rxAperture128(getNChannels(), false);
    setValuesInRange(rxAperture64, 0, 64, true);
    setValuesInRange(rxAperture128, 0, 128, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture128,
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters(),
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture128,
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters(),
        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                               (x.txAperture = getDefaultTxAperture(getNChannels()), x.rxAperture = rxAperture128,
                                x.txDelays = getDefaultTxDelays(getNChannels())))
            .getTxRxParameters()};

    EXPECT_CALL(*(us4oems[0].get()), setTxRxSequence(_, _, _, _, _, _, _, _))
        .WillOnce(Return(ByMove(createEmptySetTxRxResult(0, 6, 32))));
    EXPECT_CALL(*(us4oems[1].get()), setTxRxSequence(_, _, _, _, _, _, _, _))
        .WillOnce(Return(ByMove(createEmptySetTxRxResult(1, 6, 32))));

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
    std::optional<float> sri = std::nullopt;
    {
        testing::InSequence inSeq;
        // [1, 2]
        EXPECT_CALL(*(us4oems[0].get()), setSubsequence(2, 5, false, sri)).Times(1);
        EXPECT_CALL(*(us4oems[1].get()), setSubsequence(2, 5, false, sri)).Times(1);
        // [0, 1]
        EXPECT_CALL(*(us4oems[0].get()), setSubsequence(0, 3, false, sri)).Times(1);
        EXPECT_CALL(*(us4oems[1].get()), setSubsequence(0, 3, false, sri)).Times(1);
        // [0, 2]
        EXPECT_CALL(*(us4oems[0].get()), setSubsequence(0, 5, false, sri)).Times(1);
        EXPECT_CALL(*(us4oems[1].get()), setSubsequence(0, 5, false, sri)).Times(1);
    }
    auto [buffer0, fcm0] = probeAdapter->setSubsequence(1, 2, sri);
    auto [buffer1, fcm1] = probeAdapter->setSubsequence(0, 1, sri);
    auto [buffer2, fcm2] = probeAdapter->setSubsequence(0, 2, sri);

    // Verify

    // Buffer 0
    EXPECT_EQ(buffer0->getNumberOfElements(), 1);
    auto &element0 = buffer0->getElement(0);
    unsigned nSamples = 4096;
    EXPECT_EQ(element0.getShape(), NdArray::Shape({2 * 2 * 2 * nSamples, 32}));// 2 TX/RXs, 2 OEMs, 2 subapertures
    auto us4oemBuffer00 = buffer0->getUs4oemBuffer(0);
    auto us4oemBuffer01 = buffer0->getUs4oemBuffer(1);
    // OEM 0 layout
    NdArray::Shape expectedShape0 = {4 * nSamples, 32};
    size_t expectedSize0 = expectedShape0.product() * sizeof(int16);
    std::vector<uint16> firings;
    EXPECT_EQ(us4oemBuffer00.getNumberOfElements(), 1);
    EXPECT_EQ(us4oemBuffer00.getElement(0).getViewSize(), expectedSize0);
    EXPECT_EQ(us4oemBuffer00.getElement(0).getViewShape(), expectedShape0);
    auto parts = us4oemBuffer00.getElementParts();
    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
                   [](const auto &part) { return part.getFiring(); });
    EXPECT_EQ(firings, std::vector<uint16>({2, 3, 4, 5}));
    // OEM 1 layout
    EXPECT_EQ(us4oemBuffer01.getNumberOfElements(), 1);
    EXPECT_EQ(us4oemBuffer01.getElement(0).getViewSize(), expectedSize0);
    EXPECT_EQ(us4oemBuffer01.getElement(0).getViewShape(), expectedShape0);
    parts = us4oemBuffer01.getElementParts();
    firings.clear();
    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
                   [](const auto &part) { return part.getFiring(); });
    EXPECT_EQ(firings, std::vector<uint16>({2, 3, 4, 5}));

    // Buffer 1
    EXPECT_EQ(buffer1->getNumberOfElements(), 1);
    auto &element1 = buffer1->getElement(0);
    EXPECT_EQ(element1.getShape(), NdArray::Shape({2 * 2 * 2 * nSamples, 32}));// 2 TX/RXs, 2 OEMs, 2 subapertures
    auto us4oemBuffer10 = buffer1->getUs4oemBuffer(0);
    auto us4oemBuffer11 = buffer1->getUs4oemBuffer(1);
    // OEM 0 layout
    NdArray::Shape expectedShape1 = {4 * nSamples, 32};
    size_t expectedSize1 = expectedShape1.product() * sizeof(int16);
    EXPECT_EQ(us4oemBuffer10.getNumberOfElements(), 1);
    EXPECT_EQ(us4oemBuffer10.getElement(0).getViewSize(), expectedSize1);
    EXPECT_EQ(us4oemBuffer10.getElement(0).getViewShape(), expectedShape1);
    parts = us4oemBuffer10.getElementParts();
    firings.clear();
    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
                   [](const auto &part) { return part.getFiring(); });
    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3}));

    // OEM 1 layout
    EXPECT_EQ(us4oemBuffer11.getNumberOfElements(), 1);
    EXPECT_EQ(us4oemBuffer11.getElement(0).getViewSize(), expectedSize1);
    EXPECT_EQ(us4oemBuffer11.getElement(0).getViewShape(), expectedShape1);firings.clear();
    parts = us4oemBuffer11.getElementParts();
    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
                   [](const auto &part) { return part.getFiring(); });
    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3}));

    // Buffer 2
    EXPECT_EQ(buffer2->getNumberOfElements(), 1);
    auto &element2 = buffer2->getElement(0);
    EXPECT_EQ(element2.getShape(), NdArray::Shape({3 * 2 * 2 * nSamples, 32}));// 3 TX/RXs, 2 OEMs, 2 subapertures
    auto us4oemBuffer20 = buffer2->getUs4oemBuffer(0);
    auto us4oemBuffer21 = buffer2->getUs4oemBuffer(1);
    // OEM 0 layout
    NdArray::Shape expectedShape2 = {6 * nSamples, 32};
    size_t expectedSize2 = expectedShape2.product() * sizeof(int16);
    EXPECT_EQ(us4oemBuffer20.getNumberOfElements(), 1);
    EXPECT_EQ(us4oemBuffer20.getElement(0).getViewSize(), expectedSize2);
    EXPECT_EQ(us4oemBuffer20.getElement(0).getViewShape(), expectedShape2);
    parts = us4oemBuffer20.getElementParts();
    firings.clear();
    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
                   [](const auto &part) { return part.getFiring(); });
    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3, 4, 5}));

    // OEM 1 layout
    EXPECT_EQ(us4oemBuffer21.getNumberOfElements(), 1);
    EXPECT_EQ(us4oemBuffer21.getElement(0).getViewSize(), expectedSize2);
    EXPECT_EQ(us4oemBuffer21.getElement(0).getViewShape(), expectedShape2);
    parts = us4oemBuffer21.getElementParts();
    firings.clear();
    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
                   [](const auto &part) { return part.getFiring(); });
    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3, 4, 5}));
}

}// namespace

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
