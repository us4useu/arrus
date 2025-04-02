#include <gtest/gtest.h>

#include "Us4RImpl.h"
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

TEST(Us4RImplTest, CalculatesCorrectRxDelay) {
    std::function<float(float)> defaultTxFunc = [] (float frequency) {return frequency;};
    constexpr ChannelIdx nChannels = 192;
    std::vector<float> delays0(nChannels, 0.0f);
    std::vector<float> delays1(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);
    // Partially filled
    BitMask txAperture1(nChannels, false);
    std::fill(std::begin(txAperture1), std::begin(txAperture1) + 10, true);

    ops::us4r::Pulse pulse0{2.0e6f, 2.0f, false};
    ops::us4r::Pulse pulse1{3.0e6f, 3.0f, true};
    for (int i = 0; i < nChannels; ++i) {
        delays0[i] = i * 10e-7;
    }
    for (int i = 0; i < nChannels; ++i) {
        delays1[i] = 0.0f;
    }
    std::vector<TxRx> txrxs = {
        // Linearly increasing TX delays.
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
            )
        ).getTxRx(),
        // All TX delays the same.
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays1,
                x.pulse = pulse1
            )
        ).getTxRx(),
        // Partial TX aperture.
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture1,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse1
            )
        ).getTxRx()
    };
    TxRxSequence seq{txrxs, {}};

    float actualRxDelay0 = Us4RImpl::getRxDelay(seq.getOps().at(0), defaultTxFunc);
    float expectedRxDelay0 = *std::max_element(std::begin(delays0), std::end(delays0))
        + 1.0f / pulse0.getCenterFrequency() * pulse0.getNPeriods();
    EXPECT_EQ(expectedRxDelay0, actualRxDelay0);

    float expectedRxDelay1 = *std::max_element(std::begin(delays1), std::end(delays1))
        + 1.0f / pulse1.getCenterFrequency() * pulse1.getNPeriods();
    float actualRxDelay1 = Us4RImpl::getRxDelay(seq.getOps().at(1), defaultTxFunc);
    EXPECT_EQ(expectedRxDelay1, actualRxDelay1);


    float expectedRxDelay2 = *std::max_element(std::begin(delays0), std::begin(delays0) + 10)
        + 1.0f / pulse1.getCenterFrequency() * pulse1.getNPeriods();
    float actualRxDelay2 = Us4RImpl::getRxDelay(seq.getOps().at(2), defaultTxFunc);
    EXPECT_EQ(expectedRxDelay2, actualRxDelay2);
}

//TEST_F(A2OConverterTestMappingEsaote3, SetsSubapertureCorrectly) {
//    BitMask defaultTxAperture(nChannels, false);
//    std::vector<float> defaultDelays(nChannels, 0.0f);
//    BitMask rxAperture64(nChannels, false);
//    BitMask rxAperture128(nChannels, false);
//    setValuesInRange(rxAperture64, 0, 64, true);
//    setValuesInRange(rxAperture128, 0, 128, true);
//    std::vector<TxRxParameters> seq = {
//        ARRUS_STRUCT_INIT_LIST(
//            TestTxRxParams,
//            (
//                x.txAperture = defaultTxAperture,
//                x.rxAperture = rxAperture128,
//                x.txDelays = defaultDelays
//            )
//        ).get(),
//        ARRUS_STRUCT_INIT_LIST(
//            TestTxRxParams,
//            (
//                x.txAperture = defaultTxAperture,
//                x.rxAperture = rxAperture128,
//                x.txDelays = defaultDelays
//            )
//        ).get(),
//        ARRUS_STRUCT_INIT_LIST(
//            TestTxRxParams,
//            (
//                x.txAperture = defaultTxAperture,
//                x.rxAperture = rxAperture128,
//                x.txDelays = defaultDelays
//            )
//        ).get()
//    };
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
//    std::vector<uint16> timeoutIds;
//    EXPECT_EQ(us4oemBuffer00.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer00.getElement(0).getViewSize(), expectedSize0);
//    EXPECT_EQ(us4oemBuffer00.getElement(0).getViewShape(), expectedShape0);
//    auto parts = us4oemBuffer00.getElementParts();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(timeoutIds),
//                   [](const auto &part) { return part.getGlobalFiring(); });
//    EXPECT_EQ(timeoutIds, std::vector<uint16>({2, 3, 4, 5}));
//    // OEM 1 layout
//    EXPECT_EQ(us4oemBuffer01.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer01.getElement(0).getViewSize(), expectedSize0);
//    EXPECT_EQ(us4oemBuffer01.getElement(0).getViewShape(), expectedShape0);
//    parts = us4oemBuffer01.getElementParts();
//    timeoutIds.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(timeoutIds),
//                   [](const auto &part) { return part.getGlobalFiring(); });
//    EXPECT_EQ(timeoutIds, std::vector<uint16>({2, 3, 4, 5}));
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
//    timeoutIds.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(timeoutIds),
//                   [](const auto &part) { return part.getGlobalFiring(); });
//    EXPECT_EQ(timeoutIds, std::vector<uint16>({0, 1, 2, 3}));
//
//    // OEM 1 layout
//    EXPECT_EQ(us4oemBuffer11.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer11.getElement(0).getViewSize(), expectedSize1);
//    EXPECT_EQ(us4oemBuffer11.getElement(0).getViewShape(), expectedShape1);timeoutIds.clear();
//    parts = us4oemBuffer11.getElementParts();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(timeoutIds),
//                   [](const auto &part) { return part.getGlobalFiring(); });
//    EXPECT_EQ(timeoutIds, std::vector<uint16>({0, 1, 2, 3}));
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
//    timeoutIds.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(timeoutIds),
//                   [](const auto &part) { return part.getGlobalFiring(); });
//    EXPECT_EQ(timeoutIds, std::vector<uint16>({0, 1, 2, 3, 4, 5}));
//
//    // OEM 1 layout
//    EXPECT_EQ(us4oemBuffer21.getNumberOfElements(), 1);
//    EXPECT_EQ(us4oemBuffer21.getElement(0).getViewSize(), expectedSize2);
//    EXPECT_EQ(us4oemBuffer21.getElement(0).getViewShape(), expectedShape2);
//    parts = us4oemBuffer21.getElementParts();
//    timeoutIds.clear();
//    std::transform(std::begin(parts), std::end(parts), std::back_inserter(timeoutIds),
//                   [](const auto &part) { return part.getGlobalFiring(); });
//    EXPECT_EQ(timeoutIds, std::vector<uint16>({0, 1, 2, 3, 4, 5}));
//}


}

