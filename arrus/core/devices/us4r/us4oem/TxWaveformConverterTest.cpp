#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include "TxWaveformConverter.h"
#include "arrus/core/api/ops/us4r/Pulse.h"

namespace {
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;

struct TxWaveformConverterTestParams {
    Waveform input;
    std::vector<uint32_t> expectedOutput;
};

class TxWaveformConverterTest : public ::testing::TestWithParams<TxWaveformConverterTestParams> {};

TEST_P(TxWaveformConverterTest, ConvertsCorrectWaveform) {
    const auto &input = GetParam().input;
    const auto &expected = GetParam().expectedOutput;
    auto actual = TxWaveformConverter::toPulser(input);
    EXPECT_EQ(actual, expected);
//    EXPECT_EQ(expected.size(), actual.size());
//    for(int i = 0; i < expected.size(); ++i) {
//        EXPECT_EQ
//    }

}

INSTANTIATE_TEST_CASE_P(
    ConvertsCorrectWaveform, TxWaveformConverterTest,
    testing::Values(
        BuildSequenceWaveformParams{::arrus::ops::Pulse(1e6f, 2, false).toWaveform(), std::vector<uint16_t>{0x03F5}}, //1 firing, 1 MHz, 2 states, no repetition
//        BuildSequenceWaveformParams{8e6f, 2, 8, 0.0f, 3, std::vector<uint16_t>{0x0065}}, //8 firings, 8 MHz, 2 states (same waveform so expect 1 write), no repetition
//        BuildSequenceWaveformParams{1e6f, 8, 1, 0.0f, 3, std::vector<uint16_t>{0x53F5}}, //1 firing, 1 MHz, 8 states, 2-state repetition
//        BuildSequenceWaveformParams{8e6f, 8, 8, 0.0f, 3, std::vector<uint16_t>{0x5065}}, //8 firings, 8 MHz, 8 states (same waveform so expect 1 write), 2-state repetition
//        BuildSequenceWaveformParams{1e6f, 1, 1, 0.0f, 2, std::vector<uint16_t>{0x03F5}}, //1 firing, 1 MHz, 1 state
//        BuildSequenceWaveformParams{1e6f, 3, 1, 0.0f, 4, std::vector<uint16_t>{0x7BF5}}, //1 firing, 1 MHz, 3 state
//        BuildSequenceWaveformParams{1e6f, 2, 8, 1e6f, 3, std::vector<uint16_t>{0x03F5, 0x01F5, 0x0145, 0xE5, 0xB5, 0x95, 0x75, 0x65}}, //8 firings, 1 - 8 MHz (1 MHz step), 2 states (different waveforms so expect 8 writes)
//        BuildSequenceWaveformParams{1e6f, 1024, 1, 0.0f, 17, std::vector<uint16_t>{0xC8E5}})); //1 firing, 1 MHz, 1024 states (soft start enabled = 17 states, C8E5 = REP4STATES, + repeat + time + HVP1 pulse)
);
}