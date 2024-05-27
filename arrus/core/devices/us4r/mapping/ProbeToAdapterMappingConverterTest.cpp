#include <gtest/gtest.h>

#include "arrus/core/common/logging.h"
#include "arrus/core/devices/probe/ProbeImpl.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "ProbeToAdapterMappingConverter.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"
#include "arrus/core/common/tests.h"

namespace {

using namespace std;
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using ::arrus::devices::FrameChannelMappingAddress;

class P2AConverterTest : public ::testing::Test {
protected:
    void SetUp() override {
        float delay = 0.0f;
        for(size_t i = 0; i < probeModel.getNumberOfElements().get(0); ++i) {
            defaultTxDelays.push_back(delay);
            delay += 1e-7;
        }
    }

    ProbeToAdapterMappingConverter createConverter(
        const std::vector<ChannelIdx> &channelMapping,
        const std::unordered_set<ChannelIdx> &channelMask, ChannelIdx adapterNChannels
    ) {
        ProbeSettings probeSettings{probeModel, channelMapping};
        return ProbeToAdapterMappingConverter{
            probeId, probeId, probeSettings, probeSettings, channelMask, channelMask, adapterNChannels
        };
    }

    static TxRxParametersSequence convert(
        ProbeToAdapterMappingConverter &converter,
        const TxRxParametersSequence& inputSequence
    ) {
        auto [outputSequence, arrays] = converter.convert(0, inputSequence, {});
        return outputSequence;
    }

    static std::vector<TxRxParameters> convert(
        ProbeToAdapterMappingConverter &converter,
        const std::vector<TxRxParameters>& txrxs
    ) {
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = txrxs)).get();
        auto [outputSequence, arrays] = converter.convert(0, seq, {});
        return outputSequence.getParameters();
    }

    static FrameChannelMapping::Handle createFCM(size_t nOps, ChannelIdx nChannels) {
        std::vector<ChannelIdx> mapping = arrus::getRange<ChannelIdx>(0, nChannels);
        BitMask aperture(nChannels, true);
        return createFCM(nOps, mapping, aperture);
    }

    static FrameChannelMapping::Handle createFCM(size_t nOps, const std::vector<ChannelIdx> &mapping, const BitMask &aperture) {
        auto nChannels = aperture.size();
        auto nRxChannels = DEFAULT_DESCRIPTOR.getNRxChannels();
        FrameChannelMappingBuilder builder(nOps, nChannels);
        for(size_t op = 0; op < nOps; ++op) {
            for(size_t ch = 0; ch < nChannels; ++ch) {
                if(aperture.at(ch)) {
                    ChannelIdx mappedChannel = mapping.at(ch);
                    builder.setChannelMapping(op, ch, 0, mappedChannel/nRxChannels, mappedChannel%nRxChannels);
                }
            }
        }
        return builder.build();
    }

    void expectEqual(const FrameChannelMapping::Handle &a, const FrameChannelMapping::Handle &b) {
        EXPECT_EQ(a->getNumberOfLogicalChannels(), b->getNumberOfLogicalChannels());
        EXPECT_EQ(a->getNumberOfLogicalFrames(), b->getNumberOfLogicalFrames());
        for(size_t frame = 0; frame < a->getNumberOfLogicalFrames(); ++frame) {
            for(size_t ch = 0; ch < a->getNumberOfLogicalChannels(); ++ch) {
                auto aAddr = a->getLogical(frame, ch);
                auto bAddr = b->getLogical(frame, ch);
                EXPECT_EQ(aAddr, bAddr) << format("Failed for frame: {} channel: {}: a: {}, b: {}",
                                                  frame, ch, aAddr.toString(), bAddr.toString());
            }
        }
    }

    DeviceId probeId{arrus::devices::DeviceType::Probe, 0};
    ProbeModel probeModel{
        ProbeModelId{"test", "test"},
        Tuple<ChannelIdx>{64},
        Tuple<double>{1e-3},
        Interval<float>{1e6, 10e6},
        Interval<Voltage>{5, 90},
        0.0f
    };
    std::vector<float> defaultTxDelays;
};

TEST_F(P2AConverterTest, ProperlyCopiesAllParametersToAdapterSequence) {
    // Given
    // Channel mapping.
    auto probeNElements = probeModel.getNumberOfElements().get(0);
    std::vector<ChannelIdx> mapping = ::arrus::getRange<ChannelIdx>(0, probeNElements);
    // Input sequence.
    BitMask aperture(probeModel.getNumberOfElements().get(0), true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = aperture,
                x.rxAperture = aperture,
                x.txDelays = defaultTxDelays
            )
        )
        .get()
    };
    auto adapterNChannels = probeNElements;
    // FCM
    auto inputFCM = createFCM(1, adapterNChannels);

    auto converter = createConverter(mapping, {}, adapterNChannels);
    // Expect

    // Sequence:
    auto outputSequence = convert(converter, seq);
    EXPECT_EQ(outputSequence.size(), 1);
    auto actualTxRx = outputSequence.at(0);
    auto expectedTxRx = seq.at(0);
    EXPECT_EQ(actualTxRx, expectedTxRx);

    // FCM:
    auto outputFCM = converter.convert(inputFCM);
    expectEqual(inputFCM, outputFCM);
}

TEST_F(P2AConverterTest, ConvertsOneToOneMappingWithBiggerAdapter) {
    // Given
    // Channel mapping.
    auto probeNElements = probeModel.getNumberOfElements().get(0);
    std::vector<ChannelIdx> mapping = ::arrus::getRange<ChannelIdx>(0, probeNElements);
    // Input sequence.
    BitMask aperture(probeModel.getNumberOfElements().get(0), true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = aperture,
                x.rxAperture = aperture,
                x.txDelays = defaultTxDelays
             )
        )
        .get()
    };
    auto adapterNChannels = 2*probeNElements;
    auto inputFCM = createFCM(1, probeNElements);

    auto converter = createConverter(mapping, {}, adapterNChannels);

    // Expect:
    // Sequence:
    auto outputSequence = convert(converter, seq);
    EXPECT_EQ(outputSequence.size(), 1);
    auto actualTxRx = outputSequence.at(0);
    BitMask expectedAperture(adapterNChannels, false);
    setValuesInRange(expectedAperture, 0, probeNElements, true);
    std::vector<float> expectedDelays(adapterNChannels, 0.0f);
    for(size_t i = 0; i < defaultTxDelays.size(); ++i){
        expectedDelays[i] = defaultTxDelays[i];
    }
    EXPECT_EQ(actualTxRx.getTxAperture(), expectedAperture);
    EXPECT_EQ(actualTxRx.getRxAperture(), expectedAperture);
    EXPECT_EQ(actualTxRx.getTxDelays(), expectedDelays);
    // FCM:
    auto outputFCM = converter.convert(inputFCM);
    // The output FCM should be the same as the input one (because we are using the full RX aperture).
    expectEqual(inputFCM, outputFCM);
}

TEST_F(P2AConverterTest, ConvertsNoncontiguousMapping) {
    // Given
    // Channel mapping.
    auto probeNElements = probeModel.getNumberOfElements().get(0);
    std::vector<ChannelIdx> mappingFirstHalf = ::arrus::getRange<ChannelIdx>(0, 32);
    std::vector<ChannelIdx> mappingSecondHalf = ::arrus::getRange<ChannelIdx>(96, 128);
    std::vector<ChannelIdx> mapping(probeNElements, 0);
    std::copy(begin(mappingFirstHalf), end(mappingFirstHalf), begin(mapping));
    std::copy(begin(mappingSecondHalf), end(mappingSecondHalf), begin(mapping)+mappingFirstHalf.size());

    // Input sequence.
    BitMask aperture(probeModel.getNumberOfElements().get(0), true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = aperture,
                x.rxAperture = aperture,
                x.txDelays = defaultTxDelays
            )
        )
        .get()
    };
    auto adapterNChannels = 128;
    auto inputFCM = createFCM(1, probeNElements);

    auto converter = createConverter(mapping, {}, adapterNChannels);

    // Expect:
    // Sequence:
    auto outputSequence = convert(converter, seq);
    EXPECT_EQ(outputSequence.size(), 1);
    auto actualTxRx = outputSequence.at(0);

    BitMask expectedAperture(adapterNChannels, false);
    setValuesInRange(expectedAperture, 0, 32, true);
    setValuesInRange(expectedAperture, 96, 128, true);

    std::vector<float> expectedDelays(adapterNChannels, 0.0f);
    std::copy(begin(defaultTxDelays), begin(defaultTxDelays)+32, begin(expectedDelays));
    std::copy(begin(defaultTxDelays)+32, end(defaultTxDelays), begin(expectedDelays)+96);
    EXPECT_EQ(actualTxRx.getTxAperture(), expectedAperture);
    EXPECT_EQ(actualTxRx.getRxAperture(), expectedAperture);
    EXPECT_EQ(actualTxRx.getTxDelays(), expectedDelays);
    // FCM:
    auto outputFCM = converter.convert(inputFCM);
    // The output FCM should be the same as the input one (we are using the full RX aperture, pin mapping is one to one).
    expectEqual(inputFCM, outputFCM);
}

TEST_F(P2AConverterTest, ConvertsNonStandardMapping) {
    // Given
    // Channel mapping.
    auto probeNElements = probeModel.getNumberOfElements().get(0);
    // numpy: x = np.arange(128); np.random.shuffle(x); x[:64].tolist()
    std::vector<ChannelIdx> mapping = {
        42, 3, 78, 76, 40, 112, 8, 11, 83, 31, 70, 115, 38, 121, 0, 9, 19, 29, 10, 67, 49, 107, 102, 37, 122, 73, 126,
        15, 110, 30, 95, 13, 68, 86, 119, 17, 26, 57, 63, 108, 72, 94, 43, 23, 88, 2, 89, 91, 111, 14, 66, 93, 90, 87,
        12, 80, 116, 75, 117, 71, 36, 100, 5, 54
    };

    // Input sequence.
    BitMask aperture(probeModel.getNumberOfElements().get(0), true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = aperture,
                x.rxAperture = aperture,
                x.txDelays = defaultTxDelays
            )
        )
        .get()
    };
    auto adapterNChannels = 128;
    auto inputFCM = createFCM(1, probeNElements);

    auto converter = createConverter(mapping, {}, adapterNChannels);
    // Expect:
    // Sequence:
    auto outputSequence = convert(converter, seq);
    EXPECT_EQ(outputSequence.size(), 1);
    auto actualTxRx = outputSequence.at(0);

    BitMask expectedAperture(adapterNChannels, false);
    std::vector<float> expectedDelays(adapterNChannels, 0.0f);

    size_t i = 0;
    std::for_each(begin(mapping), end(mapping), [this, &expectedAperture, &expectedDelays, &i](const ChannelIdx &v) {
        expectedAperture.at(v) = true;
        expectedDelays.at(v) = this->defaultTxDelays.at(i);
        ++i;
    });

    EXPECT_EQ(actualTxRx.getTxAperture(), expectedAperture);
    EXPECT_EQ(actualTxRx.getRxAperture(), expectedAperture);
    EXPECT_EQ(actualTxRx.getTxDelays(), expectedDelays);
    // FCM:
    auto outputFCM = converter.convert(inputFCM);
    std::cout << "OUTPUT: " << std::endl;
    std::cout << outputFCM->toString() << std::endl;

    auto expectedFCM = createFCM(1, mapping, aperture);

    std::cout << "EXPECTED: " << std::endl;
    std::cout << expectedFCM->toString() << std::endl;
    // The output FCM should be the same as the input one (we are using the full RX aperture, pin mapping is one to one).
    expectEqual(expectedFCM, outputFCM);
}

}

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
