#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_MAPPING_ADAPTERTOUS4OEMMAPPINGCONVERTERTEST_H_
#define ARRUS_ARRUS_CORE_DEVICES_US4R_MAPPING_ADAPTERTOUS4OEMMAPPINGCONVERTERTEST_H_

#include <gtest/gtest.h>

#include "AdaterToUs4OEMMappingConverter.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/probe/ProbeImpl.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {

using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;
using ::arrus::framework::NdArray;
//using ::testing::_;
//using ::testing::AllOf;
//using ::testing::ByMove;
//using ::testing::ElementsAre;
//using ::testing::Eq;
//using ::testing::Property;
//using ::testing::Return;

class A2OConverterTest : public ::testing::Test {
protected:
    AdapterToUs4OEMMappingConverter createConverter(
        const ProbeAdapterSettings &settings, const Ordinal noems,
        const std::vector<std::vector<uint8_t>> &oemMappings
    ) {
        return AdapterToUs4OEMMappingConverter{settings, noems, oemMappings,
                                               Ordinal(0), defaultDescriptor.getNRxChannels()};
    }

    static AdapterToUs4OEMMappingConverter::SequenceByOEM convert(
        AdapterToUs4OEMMappingConverter &converter,
        const TxRxParametersSequence& inputSequence
    ) {
        auto [outputSequences, arrays] = converter.convert(0, inputSequence, {});
        return outputSequences;
    }

    static AdapterToUs4OEMMappingConverter::SequenceByOEM convert(
        AdapterToUs4OEMMappingConverter &converter,
        const std::vector<TxRxParameters>& txrxs
    ) {
        auto seq = ARRUS_STRUCT_INIT_LIST(TestTxRxParamsSequence, (x.txrx = txrxs)).get();
        auto [outputSequences, arrays] = converter.convert(0, seq, {});
        return outputSequences;
    }

    std::vector<std::vector<uint8_t>> convertToOEMMappings(
        const ProbeAdapterSettings::ChannelMapping &mapping, Ordinal nOEMs
    ) {
        std::vector<std::vector<uint8_t>> oemMapping;
        for(Ordinal o = 0; o < nOEMs; ++o) {
            std::vector<uint8_t> oem;
            std::unordered_set<uint8_t> visitedChannels;
            for(auto [oemOrdinal, ch]: mapping) {
                if(oemOrdinal == o) {
                    oem.push_back(ch);
                    visitedChannels.insert(ch);
                }
            }
            // Feel the mapping with the unused channels.
            for(uint8 i = 0; i < defaultDescriptor.getNAddressableRxChannels(); ++i) {
                if(!setContains(visitedChannels, i)) {
                    oem.push_back(i);
                }
            }
            oemMapping.push_back(oem);
        }
        return oemMapping;
    }

    Us4OEMDescriptor defaultDescriptor = DEFAULT_DESCRIPTOR;
};


class A2OConverterTestMapping1: public A2OConverterTest {
    // An adapter with 64 channels.
    // 0-32 channels to us4oem:0
    // 32-64 channels to us4oem:1
protected:
    void SetUp() override {
        Test::SetUp();

        nChannels = 64;
        ProbeAdapterSettings::ChannelMapping mapping(nChannels);
        for (ChannelIdx ch = 0; ch < nChannels; ++ch) {
            mapping[ch] = {ch / 32, ch % 32};
        }
        noems = 2;
        settings = ProbeAdapterSettings{
            ProbeAdapterModelId{"test", "test"},
            ChannelIdx(nChannels), // nChannels
            mapping
        };
        oemMappings = convertToOEMMappings(settings->getChannelMapping(), noems);
    }

    AdapterToUs4OEMMappingConverter createConverter() {
        return AdapterToUs4OEMMappingConverter{
            settings.value(), noems, oemMappings,
            Ordinal(0), defaultDescriptor.getNRxChannels()};
    }

    ChannelIdx nChannels{0};
    std::optional<ProbeAdapterSettings> settings;
    Ordinal noems{0};
    std::vector<std::vector<uint8_t>> oemMappings;
};


TEST_F(A2OConverterTestMapping1, DistributesTxAperturesCorrectly) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> txDelays(nChannels, 0.0f);
    ::arrus::setValuesInRange(txAperture, 20, 40, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = txDelays
            )
        ).get()
    };
    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    // Expected
    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 20, 32, true);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 8, true);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

TEST_F(A2OConverterTestMapping1, DistributesRxAperturesCorrectly) {
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, false);
    std::vector<float> txDelays(nChannels, 0.0f);
    ::arrus::setValuesInRange(rxAperture, 15, 51, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = txDelays
            )
            ).get()};

    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNAddressableRxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 15, 32, true);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getRxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNAddressableRxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 19, true);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getRxAperture());
}

TEST_F(A2OConverterTestMapping1, DistributesTxDelaysCorrectly) {
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);

    for (int i = 18; i < 44; ++i) {
        delays[i] = i * 5e-6;
    }
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()
    };

    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    // Expected
    std::vector<float> delays0(defaultDescriptor.getNTxChannels(), 0);
    for (int i = 18; i < 32; ++i) {
        delays0[i] = i * 5e-6;
    }
    EXPECT_EQ(delays0, sequenceOEM0.at(0).getTxDelays());

    std::vector<float> delays1(defaultDescriptor.getNTxChannels(), 0);
    for (int i = 0; i < 44 - 32; ++i) {
        delays1[i] = (i + 32) * 5e-6;
    }
    EXPECT_EQ(delays1, sequenceOEM1.at(0).getTxDelays());
}

TEST_F(A2OConverterTestMapping1, DistributesTxAperturesCorrectlySingleUs4OEM0) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);

    ::arrus::setValuesInRange(txAperture, 10, 21, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()
    };
    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 10, 21, true);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

TEST_F(A2OConverterTestMapping1, DistributesTxAperturesCorrectlySingleUs4OEM1) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);
    ::arrus::setValuesInRange(txAperture, 42, 61, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()
    };
    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNAddressableRxChannels(), false);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNAddressableRxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 10, 29, true);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

class A2OConverterTestMappingEsaote3: public A2OConverterTest {
    // An adapter with 192 channels.
    // 0-32, 64-96, 128-160 channels to us4oem:0
    // 32-64, 96-128, 160-192 channels to us4oem:1
protected:
    void SetUp() override {
        Test::SetUp();

        nChannels = 192;
        ProbeAdapterSettings::ChannelMapping mapping(nChannels);
        for (ChannelIdx ch = 0; ch < nChannels; ++ch) {
            auto group = ch / 32;
            auto module = group % 2;
            mapping[ch] = {module, ch % 32 + 32 * (group / 2)};
        }
        noems = 2;
        settings = ProbeAdapterSettings{
            ProbeAdapterModelId{"test", "test"},
            ChannelIdx(nChannels), // nChannels
            mapping
        };
        oemMappings = convertToOEMMappings(settings->getChannelMapping(), noems);
    }

    AdapterToUs4OEMMappingConverter createConverter() {
        return AdapterToUs4OEMMappingConverter{
            settings.value(), noems, oemMappings,
            Ordinal(0), defaultDescriptor.getNRxChannels()};
    }

    ChannelIdx nChannels{0};
    std::optional<ProbeAdapterSettings> settings;
    Ordinal noems{0};
    std::vector<std::vector<uint8_t>> oemMappings;
};

TEST_F(A2OConverterTestMappingEsaote3, DistributesTxAperturesCorrectlySingleUs4OEM) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);
    ::arrus::setValuesInRange(txAperture, 65, 80, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays)
            )
        .get()
    };
    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 33, 48, true);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

TEST_F(A2OConverterTestMappingEsaote3, DistributesTxAperturesCorrectlyTwoSubapertures) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);
    ::arrus::setValuesInRange(txAperture, 128 + 14, 128 + 40, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()};

    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 64 + 14, 64 + 32, true);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 64 + 0, 64 + 8, true);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

TEST_F(A2OConverterTestMappingEsaote3, DistributesTxAperturesCorrectlyThreeSubapertures) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);
    ::arrus::setValuesInRange(txAperture, 16, 80, true);
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()
    };

    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 16, 48, true);
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 32, true);
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

TEST_F(A2OConverterTestMappingEsaote3, DistributesTxAperturesWithGapsCorrectly) {
    BitMask txAperture(nChannels, false);
    BitMask rxAperture(nChannels, true);
    std::vector<float> delays(nChannels, 0.0f);
    ::arrus::setValuesInRange(txAperture, 0 + 8, 160 + 30, true);

    txAperture[0 + 14] = txAperture[0 + 17] = txAperture[32 + 23] = txAperture[32 + 24] = txAperture[64 + 25] =
        txAperture[160 + 7] = false;
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()
    };

    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 8, 96, true);
    expectedTxAp0[0 + 14] = expectedTxAp0[0 + 17] = expectedTxAp0[32 + 25] = false;
    EXPECT_EQ(expectedTxAp0, sequenceOEM0.at(0).getTxAperture());

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 64 + 30, true);
    expectedTxAp1[0 + 23] = expectedTxAp1[0 + 24] = expectedTxAp1[64 + 7] = false;
    EXPECT_EQ(expectedTxAp1, sequenceOEM1.at(0).getTxAperture());
}

TEST_F(A2OConverterTestMappingEsaote3, DistributesAperturesCorrectlyForMultipleRxApertures) {
    std::vector<float> delays(nChannels, 0.0f);
    BitMask txAperture(nChannels, false);
    ::arrus::setValuesInRange(txAperture, 0 + 8, 160 + 30, true);
    BitMask rxAperture(nChannels, false);
    ::arrus::setValuesInRange(rxAperture, 16, 96, true);
    rxAperture[0 + 18] = rxAperture[32 + 23] = false;
    // There should be two apertures: [16, 80], [80, 100] with two gaps: 18, 55

    txAperture[0 + 14] = txAperture[0 + 17] = txAperture[32 + 23] = txAperture[32 + 24] = txAperture[64 + 25] =
        txAperture[160 + 7] = false;
    std::vector<TxRxParameters> seq = {
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays
            )
        ).get()};

    auto converter = createConverter();
    auto sequenceByOEM = convert(converter, seq);
    auto sequenceOEM0 = sequenceByOEM.at(0);
    auto sequenceOEM1 = sequenceByOEM.at(1);

    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp0, 8, 96, true);
    expectedTxAp0[0 + 14] = expectedTxAp0[0 + 17] = expectedTxAp0[32 + 25] = false;

    BitMask expectedRxAp00(defaultDescriptor.getNAddressableRxChannels(), false);
    ::arrus::setValuesInRange(expectedRxAp00, 16, 32 + 80 - 64, true);
    expectedRxAp00[18] = false;
    // TODO(pjarosik) this should be done in a pretty more clever way, to minimize
    // potential transfers that are needed
    // Instead, the next one channel can be used here
    expectedRxAp00[18 + 32] = true;
    BitMask expectedRxAp01(defaultDescriptor.getNAddressableRxChannels(), false);
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

    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
    ::arrus::setValuesInRange(expectedTxAp1, 0, 64 + 30, true);
    expectedTxAp1[0 + 23] = expectedTxAp1[0 + 24] = expectedTxAp1[64 + 7] = false;

    BitMask expectedRxAp10(defaultDescriptor.getNAddressableRxChannels(), false);
    ::arrus::setValuesInRange(expectedRxAp10, 0, 32, true);
    expectedRxAp10[23] = false;

    BitMask expectedRxAp11(defaultDescriptor.getNAddressableRxChannels(), false);
    // second aperture should be empty
    EXPECT_SEQUENCE_PROPERTY_NFRAMES(1,
                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp10)),
                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp11))),
                                     2);

    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}

//TEST_F(A2OConverterTestMappingEsaote3,
//       DistributesAperturesCorrectlyForMultipleRxAperturesForFrameMetadataUs4OEM) {
//    // It should keep tx aperture on the second module even if there is no rx aperture for this module
//    BitMask txAperture(nChannels, false);
//    ::arrus::setValuesInRange(txAperture, 0 + 9, 160 + 31, true);
//
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 16, 32, true);
//    ::arrus::setValuesInRange(rxAperture, 64 + 16, 64 + 32, true);
//    rxAperture[0 + 18] = rxAperture[64 + 23] = false;
//
//    txAperture[0 + 14] = txAperture[0 + 17] = txAperture[32 + 23] = txAperture[32 + 24] = txAperture[64 + 25] =
//        txAperture[160 + 7] = false;
//    std::vector<TxRxParameters> seq = {ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
//                                                               x.txDelays = getDefaultTxDelays(nChannels)))
//                                           .getTxRxParameters()};
//    BitMask expectedTxAp0(defaultDescriptor.getNTxChannels(), false);
//    ::arrus::setValuesInRange(expectedTxAp0, 9, 96, true);
//    expectedTxAp0[0 + 14] = expectedTxAp0[0 + 17] = expectedTxAp0[32 + 25] = false;
//
//    BitMask expectedRxAp00(defaultDescriptor.getNAddressableRxChannels(), false);
//    ::arrus::setValuesInRange(expectedRxAp00, 16, 32, true);
//    expectedRxAp00[18] = false;
//    expectedRxAp00[32 + 18] = true;
//    BitMask expectedRxAp01(defaultDescriptor.getNAddressableRxChannels(), false);
//    ::arrus::setValuesInRange(expectedRxAp01, 32 + 16, 32 + 32, true);
//    expectedRxAp01[32 + 23] = false;
//    expectedRxAp01[32 + 18] = false;
//
//    EXPECT_SEQUENCE_PROPERTY_NFRAMES(0,
//                                     // Tx aperture should stay the same.
//                                     // Rx aperture should be adjusted appropriately.
//                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp0),
//                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp00)),
//                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp0),
//                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp01))),
//                                     2);
//
//    BitMask expectedTxAp1(defaultDescriptor.getNTxChannels(), false);
//    ::arrus::setValuesInRange(expectedTxAp1, 0, 64 + 31, true);
//    expectedTxAp1[0 + 23] = expectedTxAp1[0 + 24] = expectedTxAp1[64 + 7] = false;
//
//    // rx apertures should be empty
//    BitMask expectedRxAp10(defaultDescriptor.getNAddressableRxChannels(), false);
//    BitMask expectedRxAp11(defaultDescriptor.getNAddressableRxChannels(), false);
//
//    EXPECT_SEQUENCE_PROPERTY_NFRAMES(1,
//                                     ElementsAre(AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
//                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp10)),
//                                                 AllOf(Property(&TxRxParameters::getTxAperture, expectedTxAp1),
//                                                       Property(&TxRxParameters::getRxAperture, expectedRxAp11))),
//                                     2);
//
//    SET_TX_RX_SEQUENCE(probeAdapter, seq);
//}
//
//TEST_F(A2OConverterTestMappingEsaote3, DistributesTxAperturesTwoOperations) {
//    BitMask txAperture0(nChannels, false);
//    ::arrus::setValuesInRange(txAperture0, 20, 64 + 20, true);
//    BitMask txAperture1(nChannels, false);
//    ::arrus::setValuesInRange(txAperture1, 23, 64 + 23, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = txAperture0, x.rxAperture = getDefaultRxAperture(nChannels),
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters(),
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = txAperture1, x.rxAperture = getDefaultRxAperture(nChannels),
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters()};
//    BitMask expectedTxAp00(defaultDescriptor.getNTxChannels(), false);
//    ::arrus::setValuesInRange(expectedTxAp00, 20, 32 + 20, true);
//    BitMask expectedTxAp01(defaultDescriptor.getNTxChannels(), false);
//    ::arrus::setValuesInRange(expectedTxAp01, 23, 32 + 23, true);
//    EXPECT_SEQUENCE_PROPERTY_NFRAMES(0,
//                                     ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp00),
//                                                 Property(&TxRxParameters::getTxAperture, expectedTxAp01)),
//                                     2);
//
//    BitMask expectedTxAp10(defaultDescriptor.getNTxChannels(), false);
//    ::arrus::setValuesInRange(expectedTxAp10, 0, 32, true);
//    BitMask expectedTxAp11(defaultDescriptor.getNTxChannels(), false);
//    ::arrus::setValuesInRange(expectedTxAp11, 0, 32, true);
//    EXPECT_SEQUENCE_PROPERTY_NFRAMES(1,
//                                     ElementsAre(Property(&TxRxParameters::getTxAperture, expectedTxAp10),
//                                                 Property(&TxRxParameters::getTxAperture, expectedTxAp11)),
//                                     2);
//
//    SET_TX_RX_SEQUENCE(probeAdapter, seq);
//}
//
//// ------------------------------------------ Test Frame Channel Mapping
//TEST_F(A2OConverterTestMappingEsaote3, ProducesCorrectFCMSingleDistributedOperation) {
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 16, 72, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture,
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters()};
//    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0; i < 32; ++i) {
//        if (i < 24) {
//            builder0.setChannelMapping(0, i, 0, 0, i);
//        } else {
//            builder0.setChannelMapping(0, i, 0, 0, -1);
//        }
//    }
//    auto fcm0 = builder0.build();
//
//    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0; i < 32; ++i) {
//        builder1.setChannelMapping(0, i, 1, 0, i);
//    }
//    auto fcm1 = builder1.build();
//    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);
//
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));
//
//    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
//    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));
//
//    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);
//
//    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
//    EXPECT_EQ(72 - 16, fcm->getNumberOfLogicalChannels());
//
//    for (int i = 0; i < 16; ++i) {
//        auto address = fcm->getLogical(0, i);
//        EXPECT_EQ(0, address.getUs4oem());
//        EXPECT_EQ(0, address.getFrame());
//        EXPECT_EQ(i, address.getChannel());
//    }
//
//    for (int i = 16; i < 16 + 32; ++i) {
//        auto address = fcm->getLogical(0, i);
//        EXPECT_EQ(1, address.getUs4oem());
//        EXPECT_EQ(0, address.getFrame());
//        EXPECT_EQ(address.getChannel(), i - 16);
//    }
//
//    for (int i = 16 + 32; i < 56; ++i) {
//        auto address = fcm->getLogical(0, i);
//        EXPECT_EQ(0, address.getUs4oem());
//        EXPECT_EQ(0, address.getFrame());
//        EXPECT_EQ(address.getChannel(), i - 32);
//    }
//
//    // Make sure the correct frame offsets are set.
//    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
//    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
//}
//
//TEST_F(A2OConverterTestMappingEsaote3, ProducesCorrectFCMSingleDistributedOperationWithGaps) {
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 16, 73, true);
//    // Channels 20, 30 and 40 were masked for given us4oem and data is missing.
//    // Still, the input rx aperture stays as is.
//
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture,
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters()};
//    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0, j = -1; i < 32; ++i) {
//        int currentJ = -1;
//        // channels were marked by the us4oem that are missing
//        if (i != 20 - 16 && i != 30 - 16 && i <= 25) {
//            currentJ = ++j;
//        }
//        builder0.setChannelMapping(0, i, 0, 0, currentJ);
//    }
//    auto fcm0 = builder0.build();
//
//    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0, j = -1; i < 32; ++i) {
//        int currentJ = -1;
//        if (i != 40 - 32) {
//            currentJ = ++j;
//        }
//        builder1.setChannelMapping(0, i, 1, 0, currentJ);
//    }
//    auto fcm1 = builder1.build();
//
//    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);
//
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));
//
//    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
//    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));
//
//    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);
//
//    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
//    EXPECT_EQ(73 - 16, fcm->getNumberOfLogicalChannels());
//
//    std::vector<FrameChannelMapping::FrameNumber> expectedFrames;
//    std::vector<FrameChannelMapping::Us4OEMNumber> expectedUs4oems;
//    for (int i = 16; i < 32; ++i) {
//        expectedUs4oems.push_back(0);
//        expectedFrames.push_back(0);
//    }
//    for (int i = 32; i < 64; ++i) {
//        expectedUs4oems.push_back(1);
//        expectedFrames.push_back(0);
//    }
//    for (int i = 64; i < 73; ++i) {
//        expectedUs4oems.push_back(0);
//        expectedFrames.push_back(0);
//    }
//    std::vector<int8> expectedChannels = {0,  1,  2,  3,  -1, 4,  5,  6,  7,  8,  9,  10, 11, 12, -1, 13, 0,  1,  2,
//                                          3,  4,  5,  6,  7,  -1, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
//                                          21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 14, 15, 16, 17, 18, 19, 20, 21, 22};
//
//    for (int i = 0; i < 73 - 16; ++i) {
//        auto address = fcm->getLogical(0, i);
//        EXPECT_EQ(expectedUs4oems[i], address.getUs4oem());
//        EXPECT_EQ(expectedFrames[i], address.getFrame());
//        EXPECT_EQ(expectedChannels[i], address.getChannel());
//    }
//    // Make sure the correct frame offsets are set.
//    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
//    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
//}
//
//TEST_F(A2OConverterTestMappingEsaote3, ProducesCorrectFCMForMultiOpRxAperture) {
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 48, 128, true);
//    // RxNOP - the second operation on us4oem
//    // Ops: us4oem0: 32-64 (64-96), Rx NOP, us4oem1: 16-48, 48-64
//    // Channel 99 (us4oem:1 channel 32+3) is masked and data is missing.
//    // Still, the input rx aperture stays as is.
//
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture,
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters()};
//    // FCM from the us4oem:0
//    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
//    // The second op is Rx NOP.
//    for (int i = 0; i < 32; ++i) {
//        builder0.setChannelMapping(0, i, 0, 0, i);
//    }
//    auto fcm0 = builder0.build();
//
//    FrameChannelMappingBuilder builder1(2, Us4OEMImpl::N_RX_CHANNELS);
//    // First frame:
//    for (int i = 0, j = -1; i < 32; ++i) {
//        int currentJ = -1;
//        if (i != 16 + 3) {
//            currentJ = ++j;
//            builder1.setChannelMapping(0, i, 1, 0, currentJ);
//        } else {
//            builder1.setChannelMapping(0, i, 1, 0, FrameChannelMapping::UNAVAILABLE);
//        }
//    }
//    // Second frame:
//    for (int i = 0; i < 32; ++i) {
//        if (i < 16) {
//            builder1.setChannelMapping(1, i, 1, 1, i);
//        } else {
//            builder1.setChannelMapping(1, i, 1, 1, FrameChannelMapping::UNAVAILABLE);
//        }
//    }
//    auto fcm1 = builder1.build();
//
//    auto us4oemBuffer = createUs4OEMBuffer(2, 32, 4096);
//
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));
//
//    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
//    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));
//
//    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);
//
//    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
//    EXPECT_EQ(128 - 48, fcm->getNumberOfLogicalChannels());
//
//    std::vector<FrameChannelMapping::Us4OEMNumber> expectedUs4oems;
//    std::vector<FrameChannelMapping::FrameNumber> expectedFrames;
//    std::vector<int8> expectedChannels;
//
//    // Us4OEM:1, frame 0, channels 0-16
//    for (int i = 48; i < 64; ++i) {
//        expectedUs4oems.push_back(1);
//        expectedFrames.push_back(0);
//        expectedChannels.push_back(i - 48);
//    }
//    // Us4OEM:0
//    for (int i = 64; i < 96; ++i) {
//        expectedUs4oems.push_back(0);
//        expectedFrames.push_back(0);
//        expectedChannels.push_back(i - 64);
//    }
//    // Us4OEM:1, frame 0, channels 16-32
//    for (int i = 96; i < 96 + 15; ++i) {// 15 because there will be one -1
//        expectedUs4oems.push_back(1);
//        expectedFrames.push_back(0);
//        if (i == 99 && expectedChannels[expectedChannels.size() - 1] != FrameChannelMapping::UNAVAILABLE) {
//            expectedChannels.push_back(FrameChannelMapping::UNAVAILABLE);
//            --i;
//        } else {
//            expectedChannels.push_back(i - 96 + 16);
//        }
//    }
//    // Us4OEM:1, frame 1
//    for (int i = 96 + 16; i < 128; ++i) {
//        expectedUs4oems.push_back(1);
//        expectedFrames.push_back(1);
//        expectedChannels.push_back(i - (96 + 16));
//    }
//
//    // VALIDATE
//    for (int i = 0; i < 128 - 48; ++i) {
//        auto address = fcm->getLogical(0, i);
//        EXPECT_EQ(expectedUs4oems[i], address.getUs4oem());
//        EXPECT_EQ(expectedFrames[i], address.getFrame());
//        EXPECT_EQ(expectedChannels[i], address.getChannel());
//    }
//    // Make sure the correct frame offsets are set.
//    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
//    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
//}
//
//// Currently padding impacts the output frame channel mapping
//TEST_F(A2OConverterTestMappingEsaote3, AppliesPaddingToFCMCorrectly) {
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 0, 16, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture,
//                                x.txDelays = getDefaultTxDelays(nChannels), x.rxPadding = {16, 0}))
//            .getTxRxParameters()};
//    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0; i < 32; ++i) {
//        if (i < 16) {
//            builder0.setChannelMapping(0, i, 0, 0, i);
//        } else {
//            builder0.setChannelMapping(0, i, 0, 0, -1);
//        }
//    }
//    auto fcm0 = builder0.build();
//
//    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
//    // No active channels
//    auto fcm1 = builder1.build();
//
//    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);
//
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));
//
//    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
//    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));
//
//    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);
//
//    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
//    EXPECT_EQ(32, fcm->getNumberOfLogicalChannels());// 16 active + 16 rx padding
//
//    for (int i = 0; i < 16; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(0, address.getFrame());
//        ASSERT_EQ(address.getChannel(), FrameChannelMapping::UNAVAILABLE);
//    }
//
//    for (int i = 16; i < 32; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(0, address.getUs4oem());
//        ASSERT_EQ(0, address.getFrame());
//        ASSERT_EQ(address.getChannel(), i - 16);
//    }
//    // Make sure the correct frame offsets are set.
//    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
//}
//
//// The same as above, but with aperture using two modules
//TEST_F(A2OConverterTestMappingEsaote3, AppliesPaddingToFCMCorrectlyRxApertureUsingTwoModules) {
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 0, 49, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture,
//                                x.txDelays = getDefaultTxDelays(nChannels), x.rxPadding = {15, 0}))
//            .getTxRxParameters()};
//    FrameChannelMappingBuilder builder0(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0; i < 32; ++i) {
//        builder0.setChannelMapping(0, i, 0, 0, i);
//    }
//    auto fcm0 = builder0.build();
//
//    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0; i < 32; ++i) {
//        if (i < 17) {
//            builder1.setChannelMapping(0, i, 1, 0, i);
//        } else {
//            builder1.setChannelMapping(0, i, 1, 0, FrameChannelMapping::UNAVAILABLE);
//        }
//    }
//    auto fcm1 = builder1.build();
//
//    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);
//
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));
//
//    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
//    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));
//
//    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);
//
//    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
//    EXPECT_EQ(64, fcm->getNumberOfLogicalChannels());// 49 active + 15 rx padding
//
//    for (int i = 0; i < 15; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(address.getChannel(), FrameChannelMapping::UNAVAILABLE);
//    }
//    for (int i = 15; i < 15 + 32; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(0, address.getUs4oem());
//        ASSERT_EQ(0, address.getFrame());
//        ASSERT_EQ(address.getChannel(), i - 15);
//    }
//    for (int i = 15 + 32; i < 64; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(1, address.getUs4oem());
//        ASSERT_EQ(0, address.getFrame());
//        ASSERT_EQ(address.getChannel(), i - (15 + 32));
//    }
//    EXPECT_EQ(0, fcm->getFirstFrame(0));// Us4OEM:0
//    EXPECT_EQ(1, fcm->getFirstFrame(1));// Us4OEM:1
//}
//
//TEST_F(A2OConverterTestMappingEsaote3, AppliesPaddingToFCMCorrectlyRightSide) {
//    BitMask rxAperture(nChannels, false);
//    ::arrus::setValuesInRange(rxAperture, 176, 192, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture,
//                                x.txDelays = getDefaultTxDelays(nChannels), x.rxPadding = {0, 16}))
//            .getTxRxParameters()};
//    FrameChannelMappingBuilder builder0(0, Us4OEMImpl::N_RX_CHANNELS);
//    // No output
//    auto fcm0 = builder0.build();
//
//    FrameChannelMappingBuilder builder1(1, Us4OEMImpl::N_RX_CHANNELS);
//    for (int i = 0; i < 32; ++i) {
//        if (i < 16) {
//            builder1.setChannelMapping(0, i, 1, 0, i);
//        } else {
//            builder1.setChannelMapping(0, i, 1, 0, FrameChannelMapping::UNAVAILABLE);
//        }
//    }
//    auto fcm1 = builder1.build();
//
//    auto us4oemBuffer = createUs4OEMBuffer(1, 32, 4096);
//
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res0(us4oemBuffer, std::move(fcm0));
//    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle> res1(us4oemBuffer, std::move(fcm1));
//
//    EXPECT_CALL(*(us4oems[0].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res0))));
//    EXPECT_CALL(*(us4oems[1].get()), US4OEM_MOCK_SET_TX_RX_SEQUENCE()).WillOnce(Return(ByMove(std::move(res1))));
//
//    auto [buffer, fcm] = SET_TX_RX_SEQUENCE(probeAdapter, seq);
//
//    EXPECT_EQ(1, fcm->getNumberOfLogicalFrames());
//    EXPECT_EQ(32, fcm->getNumberOfLogicalChannels());// 16 active + 16 rx padding
//
//    for (int i = 0; i < 16; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(1, address.getUs4oem());
//        ASSERT_EQ(0, address.getFrame());
//        ASSERT_EQ(address.getChannel(), i);
//    }
//    for (int i = 16; i < 32; ++i) {
//        auto address = fcm->getLogical(0, i);
//        ASSERT_EQ(address.getChannel(), FrameChannelMapping::UNAVAILABLE);
//    }
//    EXPECT_EQ(0, fcm->getFirstFrame(1));// Us4OEM:1
//}
//

//TEST_F(A2OConverterTestMappingEsaote3, SetsSubapertureCorrectly) {
//    BitMask rxAperture64(nChannels, false);
//    BitMask rxAperture128(nChannels, false);
//    setValuesInRange(rxAperture64, 0, 64, true);
//    setValuesInRange(rxAperture128, 0, 128, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture128,
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters(),
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture128,
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters(),
//        ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
//                               (x.txAperture = getDefaultTxAperture(nChannels), x.rxAperture = rxAperture128,
//                                x.txDelays = getDefaultTxDelays(nChannels)))
//            .getTxRxParameters()};
//
//    EXPECT_CALL(*(us4oems[0].get()), setTxRxSequence(_, _, _, _, _, _, _, _))
//        .WillOnce(Return(ByMove(createEmptySetTxRxResult(0, 6, 32))));
//    EXPECT_CALL(*(us4oems[1].get()), setTxRxSequence(_, _, _, _, _, _, _, _))
//        .WillOnce(Return(ByMove(createEmptySetTxRxResult(1, 6, 32))));
//
//    SET_TX_RX_SEQUENCE(probeAdapter, seq);
//    std::optional<float> sri = std::nullopt;
//    {
//        testing::InSequence inSeq;
//        // [1, 2]
//        EXPECT_CALL(*(us4oems[0].get()), setSubsequence(2, 5, false, sri)).Times(1);
//        EXPECT_CALL(*(us4oems[1].get()), setSubsequence(2, 5, false, sri)).Times(1);
//        // [0, 1]
//        EXPECT_CALL(*(us4oems[0].get()), setSubsequence(0, 3, false, sri)).Times(1);
//        EXPECT_CALL(*(us4oems[1].get()), setSubsequence(0, 3, false, sri)).Times(1);
//        // [0, 2]
//        EXPECT_CALL(*(us4oems[0].get()), setSubsequence(0, 5, false, sri)).Times(1);
//        EXPECT_CALL(*(us4oems[1].get()), setSubsequence(0, 5, false, sri)).Times(1);
//    }
//    auto [buffer0, fcm0] = probeAdapter->setSubsequence(1, 2, sri);
//    auto [buffer1, fcm1] = probeAdapter->setSubsequence(0, 1, sri);
//    auto [buffer2, fcm2] = probeAdapter->setSubsequence(0, 2, sri);
//
//    // Verify
//
//    // Buffer 0
//    EXPECT_EQ(buffer0->getNumberOfElements(), 1);
//    auto &element0 = buffer0->getElement(0);
//    unsigned nSamples = 4096;
//    EXPECT_EQ(element0.getShape(), NdArray::Shape({2 * 2 * 2 * nSamples, 32}));// 2 TX/RXs, 2 OEMs, 2 subapertures
//    auto us4oemBuffer00 = buffer0->getUs4oemBuffer(0);
//    auto us4oemBuffer01 = buffer0->getUs4oemBuffer(1);
//    // OEM 0 layout
//    NdArray::Shape expectedShape0 = {4 * nSamples, 32};
//    size_t expectedSize0 = expectedShape0.product() * sizeof(int16);
//    std::vector<uint16> firings;
//    EXPECT_EQ(us4oemBuffer00.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer00.getElement(0).getViewSize(), expectedSize0);
//    EXPECT_EQ(us4oemBuffer00.getElement(0).getViewShape(), expectedShape0);
//    auto parts = us4oemBuffer00.getElementParts();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
//                   [](const auto &part) { return part.getFiring(); });
//    EXPECT_EQ(firings, std::vector<uint16>({2, 3, 4, 5}));
//    // OEM 1 layout
//    EXPECT_EQ(us4oemBuffer01.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer01.getElement(0).getViewSize(), expectedSize0);
//    EXPECT_EQ(us4oemBuffer01.getElement(0).getViewShape(), expectedShape0);
//    parts = us4oemBuffer01.getElementParts();
//    firings.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
//                   [](const auto &part) { return part.getFiring(); });
//    EXPECT_EQ(firings, std::vector<uint16>({2, 3, 4, 5}));
//
//    // Buffer 1
//    EXPECT_EQ(buffer1->getNumberOfElements(), 1);
//    auto &element1 = buffer1->getElement(0);
//    EXPECT_EQ(element1.getShape(), NdArray::Shape({2 * 2 * 2 * nSamples, 32}));// 2 TX/RXs, 2 OEMs, 2 subapertures
//    auto us4oemBuffer10 = buffer1->getUs4oemBuffer(0);
//    auto us4oemBuffer11 = buffer1->getUs4oemBuffer(1);
//    // OEM 0 layout
//    NdArray::Shape expectedShape1 = {4 * nSamples, 32};
//    size_t expectedSize1 = expectedShape1.product() * sizeof(int16);
//    EXPECT_EQ(us4oemBuffer10.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer10.getElement(0).getViewSize(), expectedSize1);
//    EXPECT_EQ(us4oemBuffer10.getElement(0).getViewShape(), expectedShape1);
//    parts = us4oemBuffer10.getElementParts();
//    firings.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
//                   [](const auto &part) { return part.getFiring(); });
//    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3}));
//
//    // OEM 1 layout
//    EXPECT_EQ(us4oemBuffer11.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer11.getElement(0).getViewSize(), expectedSize1);
//    EXPECT_EQ(us4oemBuffer11.getElement(0).getViewShape(), expectedShape1);firings.clear();
//    parts = us4oemBuffer11.getElementParts();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
//                   [](const auto &part) { return part.getFiring(); });
//    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3}));
//
//    // Buffer 2
//    EXPECT_EQ(buffer2->getNumberOfElements(), 1);
//    auto &element2 = buffer2->getElement(0);
//    EXPECT_EQ(element2.getShape(), NdArray::Shape({3 * 2 * 2 * nSamples, 32}));// 3 TX/RXs, 2 OEMs, 2 subapertures
//    auto us4oemBuffer20 = buffer2->getUs4oemBuffer(0);
//    auto us4oemBuffer21 = buffer2->getUs4oemBuffer(1);
//    // OEM 0 layout
//    NdArray::Shape expectedShape2 = {6 * nSamples, 32};
//    size_t expectedSize2 = expectedShape2.product() * sizeof(int16);
//    EXPECT_EQ(us4oemBuffer20.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer20.getElement(0).getViewSize(), expectedSize2);
//    EXPECT_EQ(us4oemBuffer20.getElement(0).getViewShape(), expectedShape2);
//    parts = us4oemBuffer20.getElementParts();
//    firings.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
//                   [](const auto &part) { return part.getFiring(); });
//    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3, 4, 5}));
//
//    // OEM 1 layout
//    EXPECT_EQ(us4oemBuffer21.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer21.getElement(0).getViewSize(), expectedSize2);
//    EXPECT_EQ(us4oemBuffer21.getElement(0).getViewShape(), expectedShape2);
//    parts = us4oemBuffer21.getElementParts();
//    firings.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(firings),
//                   [](const auto &part) { return part.getFiring(); });
//    EXPECT_EQ(firings, std::vector<uint16>({0, 1, 2, 3, 4, 5}));
//}

}

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_MAPPING_ADAPTERTOUS4OEMMAPPINGCONVERTERTEST_H_
