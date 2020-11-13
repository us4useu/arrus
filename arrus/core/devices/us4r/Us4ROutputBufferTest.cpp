#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <limits>
#include <random>

#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/api/common/types.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"

namespace {

using namespace arrus;
using namespace arrus::devices;

struct TestCase {
    TestCase(Ordinal nus4Oems, uint16 nFrames, uint16 numberOfQueueElements,
             double producerSleepTimeDispersion, double consumerSleepTimeDispersion)
        : nus4oems(nus4Oems), nFrames(nFrames),
          numberOfQueueElements(numberOfQueueElements),
          producerSleepTimeDispersion(producerSleepTimeDispersion),
          consumerSleepTimeDispersion(consumerSleepTimeDispersion) {}

    Ordinal nus4oems = 1;
    uint16 nFrames = 10;
    uint16 numberOfQueueElements = 3;
    double producerSleepTimeDispersion = 0;
    double consumerSleepTimeDispersion = 0;
};

class Us4ROutputBufferTest
    : public testing::TestWithParam<TestCase> {
};

TEST_P(Us4ROutputBufferTest, TestSingleConsumerMultipleProducersWithCallback) {
    // Tests
    constexpr uint16 LOG_FREQ = 1000;
    Ordinal nus4oems = GetParam().nus4oems;
    uint16 nFrames = GetParam().nFrames;
    uint16 nElements = GetParam().numberOfQueueElements;
    constexpr uint32 N_SAMPLES = 64;
    constexpr size_t OUTPUT_SIZE = N_SAMPLES * 32; //in number of array elements

    std::vector<size_t> outputSizes = getNTimes<size_t>(N_SAMPLES, nus4oems);
    Us4ROutputBuffer buffer(outputSizes, nElements);

    // rng for time sleeps
    double producerSleepTimeDisp = GetParam().producerSleepTimeDispersion;
    double consumerSleepTimeDisp = GetParam().consumerSleepTimeDispersion;

    auto callback = [&](Ordinal n) {
        std::random_device rd{};
        std::mt19937 rng{rd()};
        std::normal_distribution<> rDistr{0.0, 1.0};
        uint16 outputNumber = 0;
        while(outputNumber < nFrames) {
            if(producerSleepTimeDisp > 0.0) {
                uint32 sleepTimeMs = std::abs(rDistr(rng)) * producerSleepTimeDisp * 1000;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeMs));
            }
            try {
                if(n == 0 && outputNumber % LOG_FREQ == 0) {
                    getDefaultLogger()->log(
                        arrus::LogSeverity::DEBUG,
                        ::arrus::format("Saving frame {}", outputNumber));
                }
                auto func = [&] {
                    uint16 *data = buffer.getAddress(
                        (outputNumber % nElements), n);
                    for(size_t i = 0; i < OUTPUT_SIZE; ++i) {
                        data[i] = outputNumber;
                    }
                    ++outputNumber;
                };
                buffer.signal(n, 10000, func);
            } catch(const std::exception &e) {
                std::cerr << e.what() << std::endl;
            }
        }
    };

    // run threads
    std::vector<std::thread> us4oems(nus4oems);
    for(int i = 0; i < nus4oems; ++i) {
        us4oems[i] = std::thread(callback, static_cast<Ordinal>(i));
    }

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<> rDistr{0.0, 1.0};
    uint16 frameNumber = 0;
    while(frameNumber < nFrames) {
        if(consumerSleepTimeDisp > 0.0) {
            uint32 sleepTimeMs = std::abs(rDistr(rng)) * consumerSleepTimeDisp * 1000;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeMs));
        }
        uint16 *d = buffer.front();
        size_t size = buffer.getElementSize();
        if(frameNumber % LOG_FREQ == 0) {
            getDefaultLogger()->log(arrus::LogSeverity::DEBUG,
                                    ::arrus::format(
                                        "Reading frame {}, size {}",
                                        frameNumber, size));
        }
        for(size_t i = 0; i < size; ++i) {
            ASSERT_EQ(d[i], frameNumber);
        }
        ++frameNumber;
        buffer.releaseFront();
    }

    for(std::thread &us4oem: us4oems) {
        us4oem.join();
    }
}
}
#define TEST_CASE_PARAMETERS_SET1(nus4oems) \
    TestCase(nus4oems, 20, 7, 0.1, 0), \
    TestCase(nus4oems, 20, 7, 0, 0.1),       \
    TestCase(nus4oems, 20, 7, 0.1, 0.1),  \
    TestCase(nus4oems, 1, 2, 0, 0), \
    TestCase(nus4oems, 100, 2, 0, 0), \
    TestCase(nus4oems, 10, 100, 0, 0), \
    TestCase(nus4oems, 100, 10, 0, 0),  \
    TestCase(nus4oems, 10000, 2, 0, 0)

INSTANTIATE_TEST_CASE_P
(SingleProducerCallback, Us4ROutputBufferTest,
 testing::Values(
     TEST_CASE_PARAMETERS_SET1(1)
 ));

INSTANTIATE_TEST_CASE_P
(TwoProducersCallback, Us4ROutputBufferTest,
 testing::Values(
     TEST_CASE_PARAMETERS_SET1(2)
 ));

INSTANTIATE_TEST_CASE_P
(ThreeProducersCallback, Us4ROutputBufferTest,
 testing::Values(
     TEST_CASE_PARAMETERS_SET1(3)
 ));

INSTANTIATE_TEST_CASE_P
(FourProducersCallback, Us4ROutputBufferTest,
 testing::Values(
     TEST_CASE_PARAMETERS_SET1(4)
 ));

INSTANTIATE_TEST_CASE_P
(EightProducersCallback, Us4ROutputBufferTest,
 testing::Values(
     TEST_CASE_PARAMETERS_SET1(8)
));

INSTANTIATE_TEST_CASE_P
(SixteenProducersCallback, Us4ROutputBufferTest,
 testing::Values(
     TEST_CASE_PARAMETERS_SET1(16)
));

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG_LEVEL(arrus::Logging, LogSeverity::DEBUG);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


