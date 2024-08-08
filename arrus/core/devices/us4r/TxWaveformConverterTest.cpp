#include <gtest/gtest.h>
#include <iostream>
#include "TxWaveformConverter.h"
#include "arrus/core/api/ops/us4r/Pulse.h"

namespace {
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;

constexpr auto ENDSTATE = TxWaveformConverter::END_STATE;

struct TxWaveformConverterTestParams {
    Waveform input;
    std::vector<uint32_t> expectedOutput;
};

TEST(TestTest, TestTestTest) {
    EXPECT_TRUE(true);
}


class TxWaveformConverterTest : public ::testing::TestWithParam<TxWaveformConverterTestParams> {};

TEST_P(TxWaveformConverterTest, ConvertsCorrectWaveform) {
    const auto &input = GetParam().input;
    const auto &expected = GetParam().expectedOutput;
    auto actual = TxWaveformConverter::toPulser(input);
    EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ConvertsCorrectWaveform, TxWaveformConverterTest,
    testing::Values(
        // NOTE: below test cases implicitly tests also Pulse -> Waveform conversion
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(1e6f, 1, false, 1).toWaveform(),
            // HVP0 -> HVM0, 1MHz @ 130 MHz -> 1 us -> 0.5us up then 0.5us down -> 65 cycles (-2) -> durationClk: 63, no repetition
            std::vector<uint32_t>{0b00000'0111111'0101, 0b00000'0111111'1010, ENDSTATE}
        },
        // Repeat
        TxWaveformConverterTestParams{
            // The same as above, only the number of repetitions is > 1
            // 2-state repetition
            // NOTE: the actual number of repetitions is 0 + 2!
            ::arrus::ops::us4r::Pulse(1e6f, 2, false, 1).toWaveform(),
            std::vector<uint32_t>{0b01000'0111111'0101, 0b00000'0111111'1010, ENDSTATE}
        },
        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // 2^8 + 1 == 257
            // ODD number of repetitions
            // This case can should be handled by 257 repetitions of the 2-state waveform
            ::arrus::ops::us4r::Pulse(1e6f, 257, false, 1).toWaveform(),
            // repeat 4-state
            std::vector<uint32_t>{0b01111'0111111'0101, 0b11111'0111111'1010, ENDSTATE}
        },
        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enoguh
            // 2^8 + 2 == 258
            // EVEN number of repetitions
            // Currently this should automatically translate to the 4-state repetition waveform.
            ::arrus::ops::us4r::Pulse(1e6f, 258, false, 1).toWaveform(),
            // repeat 4-states 129 times
            std::vector<uint32_t>{0b11010'0111111'0101, 0b10000'0111111'1010, 0b00001'0111111'0101, 0b00000'0111111'1010, ENDSTATE}
        },
        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // 2^8 + 3 == 259
            // ODD number of repetitions

            // NOTE: 259 repetitions can be done in one of the following two ways:
            // 258 cycles by 4-state repetition (4 entries) + 1 cycle with no repetition  (2 entries) = 6 entries
            // 257 cycles by 4-state repetition (2 entries) + 2 cycles by 2-state repetition (2 entries) = 4 entries
            // We choose the one minimizing the number of entries.
            ::arrus::ops::us4r::Pulse(1e6f, 259, false, 1).toWaveform(),
            // repeat 4-state
            // TODO update
            std::vector<uint32_t>{0b11000'0111111'0101, 0b00000'0111111'1010, ENDSTATE}
        },
        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // ODD number of repetitions
            // 515 = 256 (max first two entries) + 257 (max second two entries)  + 2 (see the previous example)
            // This should be handled by 6-entries waveform.
            // We choose the one minimizing the number of entries.
            ::arrus::ops::us4r::Pulse(1e6f, 259, false, 1).toWaveform(),
            // repeat 4-state
            // TODO update
            std::vector<uint32_t>{0b11000'0111111'0101, 0b00000'0111111'1010, ENDSTATE}
        },
        // level
        // other frequencies
        TxWaveformConverterTestParams{
            // 1 MHz, 1 cycle, level 0
            ::arrus::ops::us4r::Pulse(8e6f, 1, false, 1).toWaveform(),
            // HVP0 -> HVM0, 8MHz @ 130 MHz -> 1/8 us -> 1/16 us up then 1/16 us down -> ~8 cycles (-2) -> durationClk: 6, no repetition
            std::vector<uint32_t>{0b00000'0000110'0101, 0b00000'0000110'1010, ENDSTATE}
        }
//        BuildSequenceWaveformParams{1e6f, 8, 1, 0.0f, 3, std::vector<uint16_t>{0x53F5}}, //1 firing, 1 MHz, 8 states, 2-state repetition
//        BuildSequenceWaveformParams{8e6f, 8, 8, 0.0f, 3, std::vector<uint16_t>{0x5065}}, //8 firings, 8 MHz, 8 states (same waveform so expect 1 write), 2-state repetition
//        BuildSequenceWaveformParams{1e6f, 1, 1, 0.0f, 2, std::vector<uint16_t>{0x03F5}}, //1 firing, 1 MHz, 1 state
//        BuildSequenceWaveformParams{1e6f, 3, 1, 0.0f, 4, std::vector<uint16_t>{0x7BF5}}, //1 firing, 1 MHz, 3 state
//        BuildSequenceWaveformParams{1e6f, 2, 8, 1e6f, 3, std::vector<uint16_t>{0x03F5, 0x01F5, 0x0145, 0xE5, 0xB5, 0x95, 0x75, 0x65}}, //8 firings, 1 - 8 MHz (1 MHz step), 2 states (different waveforms so expect 8 writes)
//        BuildSequenceWaveformParams{1e6f, 1024, 1, 0.0f, 17, std::vector<uint16_t>{0xC8E5}})); //1 firing, 1 MHz, 1024 states (soft start enabled = 17 states, C8E5 = REP4STATES, + repeat + time + HVP1 pulse)
        )
    );
}
