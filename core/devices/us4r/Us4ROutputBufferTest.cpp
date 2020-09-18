#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <chrono>

#include "arrus/core/common/collections.h"
#include "arrus/core/api/common/types.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"

namespace {

using namespace arrus;
using namespace arrus::devices;

TEST(Us4OutputBufferTest, Us4OutputBufferTest1) {
    // TODO random sleep for producers/consummer
    // TODO parametrize
    constexpr uint16 LOG_FREQ = 100;
    constexpr Ordinal N_US4OEMS = 8;
    constexpr uint16 NUMBER_OF_FRAMES = 6000;
    constexpr uint16 NUMBER_OF_ELEMENTS = 3;
    constexpr uint32 N_SAMPLES = 64;
    constexpr size_t OUTPUT_SIZE = N_SAMPLES * 32; //in number of array elements

    std::vector<size_t> outputSizes = getNTimes<size_t>(N_SAMPLES, N_US4OEMS);
    Us4ROutputBuffer buffer(outputSizes, NUMBER_OF_ELEMENTS);

    auto callback = [&](Ordinal n) {
        uint16 outputNumber = 0;
        while(outputNumber < NUMBER_OF_FRAMES) {
            try {
                if(n == 0 && outputNumber % LOG_FREQ == 0) {
                    getDefaultLogger()->log(
                        arrus::LogSeverity::DEBUG,
                        ::arrus::format("Saving frame {}", outputNumber));
                }

                // should block, when the data is not ready -
                // writing to data should happen in synchronized way, as it will
                // be done by us4oems
                // the other option is to execute transfer after
                // unblocking the thread, when the data is ready
                auto func = [&] {
                    uint16 *data = buffer.getAddress(
                        (outputNumber % NUMBER_OF_ELEMENTS), n);
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
    std::vector<std::thread> us4oems(N_US4OEMS);
    for(int i = 0; i < N_US4OEMS; ++i) {
        us4oems[i] = std::thread(callback, static_cast<Ordinal>(i));
    }

    uint16 frameNumber = 0;
    while(frameNumber < NUMBER_OF_FRAMES) {
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

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG_LEVEL(arrus::Logging, LogSeverity::DEBUG);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


