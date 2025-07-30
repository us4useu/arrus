#include <gtest/gtest.h>
#include <iostream>
#include "TxWaveformConverter.h"
#include "arrus/core/api/ops/us4r/Pulse.h"

namespace {
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;

constexpr auto ENDSTATE = TxWaveformConverter::END_STATE;
constexpr auto SAMPLING_FREQUENCY = 130.0e6f;

struct TxWaveformConverterTestParams {
    Waveform input;
    std::vector<uint32_t> expectedOutput;
};


class TxWaveformConverterTest : public ::testing::TestWithParam<TxWaveformConverterTestParams> {};

TEST_P(TxWaveformConverterTest, ConvertsCorrectWaveform) {
    const auto &input = GetParam().input;
    const auto &expected = GetParam().expectedOutput;
    auto actual = TxWaveformConverter::toPulser(input);
    EXPECT_EQ(actual, expected);
}

using State = TxWaveformConverter::DeviceState;

INSTANTIATE_TEST_SUITE_P(
    ConvertsCorrectWaveform, TxWaveformConverterTest,
    testing::Values(
        // 0
        // NOTE: below test cases implicitly tests also Pulse -> Waveform conversion
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(1e6f, 1, false, 2).toWaveform(),
////             HVP0 -> HVM0, 1MHz @ 130 MHz -> 1 us -> 0.5us up then 0.5us down -> 65 cycles (-2) -> durationClk: 63, no repetition
            std::vector<uint32_t>{0b00000'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0, ENDSTATE}

        },

        // 1

//        // Reptitions
        TxWaveformConverterTestParams{
//             The same as above, only the number of repetitions is > 1
//             2-state repetition
//             NOTE: the actual number of repetitions is 0 + 2!
            ::arrus::ops::us4r::Pulse(1e6f, 2, false, 2).toWaveform(),
            std::vector<uint32_t>{0b01000'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0, ENDSTATE}
        },

        // 2

        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // 2^8 + 1 == 257
            // ODD number of repetitions
            // This case can should be handled by 257 repetitions of the 2-state waveform
            ::arrus::ops::us4r::Pulse(1e6f, 257, false, 2).toWaveform(),
            // repeat 4-state
            std::vector<uint32_t>{0b01111'0111111'0000 | State::HVP0, 0b11111'0111111'0000 | State::HVM0, ENDSTATE}
        },

        // 3

        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // 2^8 + 2 == 258
            // EVEN number of repetitions
            // Currently this should automatically translate to the 4-state repetition waveform.
            ::arrus::ops::us4r::Pulse(1e6f, 258, false, 2).toWaveform(),
            // repeat 4-states 129 times
            // 1111'0111111'0000 | State::HVM0
            std::vector<uint32_t>{0b11111'0111111'0000 | State::HVP0, 0b01111'0111111'0000 | State::HVM0, 0b00000'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0, ENDSTATE}
        },

        // 4

        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // 2^8 + 3 == 259
            // ODD number of repetitions

            // NOTE: 259 repetitions can be done in one of the following two ways:
            // 258 cycles by 4-state repetition (4 entries) + 1 cycle with no repetition  (2 entries) = 6 entries
            // 257 cycles by 2-state repetition (2 entries) + 2 cycles by 2-state repetition (2 entries) = 4 entries
            // We choose the one minimizing the number of entries.
            ::arrus::ops::us4r::Pulse(1e6f, 259, false, 2).toWaveform(),
            //                    257 repetitions of 2-cycle                  // 2 cycles, no repeat
            std::vector<uint32_t>{0b01111'0111111'0000 | State::HVP0, 0b11111'0111111'0000 | State::HVM0, 0b01000'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0,  ENDSTATE}
        },

        // 5

        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // 2^8 + 3 == 261
            // ODD number of repetitions
            ::arrus::ops::us4r::Pulse(1e6f, 261, false, 2).toWaveform(),
            //                    257 repetitions of 2-cycle                  // 2 cycles, 1 repeat (2x)
            std::vector<uint32_t>{0b01111'0111111'0000 | State::HVP0, 0b11111'0111111'0000 | State::HVM0, 0b01010'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0,  ENDSTATE}
        },

        // 6

        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // ODD number of repetitions
            // 513 = 257 (max first two entries) + 256 (max second two entries - 1)
            // This should be handled by 6-entries waveform.
            // We choose the one minimizing the number of entries.
            ::arrus::ops::us4r::Pulse(1e6f, 513, false, 2).toWaveform(),
            // repeat 4-state
            std::vector<uint32_t>{0b01111'0111111'0000 | State::HVP0, 0b11111'0111111'0000 | State::HVM0, 0b01110'0111111'0000 | State::HVP0, 0b11111'0111111'0000 | State::HVM0, ENDSTATE}
        },

        // 7

        TxWaveformConverterTestParams{
            // Test the possibility to transmit with longer pulses, in case the number of cycles is large enough
            // ODD number of repetitions
            // This should be handled by 6-entries waveform (4 repeat state + 1 to complete od number).
            // We choose the one minimizing the number of entries.
            ::arrus::ops::us4r::Pulse(1e6f, 515, false, 2).toWaveform(),
            //                    repeat 4-state 257 times                                                                2 states with no repetition
            std::vector<uint32_t>{0b11111'0111111'0000 | State::HVP0, 0b11111'0111111'0000 | State::HVM0, 0b00000'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0, 0b00000'0111111'0000 | State::HVP0, 0b00000'0111111'0000 | State::HVM0, ENDSTATE}
        },

        // 8

        // other frequencies
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(8e6f, 1, false, 2).toWaveform(),
            // HVP0 -> HVM0, 8MHz @ 130 MHz -> 1/8 us -> 1/16 us up then 1/16 us down -> ~8 cycles (-2) -> durationClk: 6, no repetition
            std::vector<uint32_t>{0b00000'0000110'0000 | State::HVP0, 0b00000'0000110'0000 | State::HVM0, ENDSTATE}
        },

        // 9

        // levels
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(8e6f, 1, false, 1).toWaveform(),
            std::vector<uint32_t>{0b00000'0000110'0000 | State::HVP1, 0b00000'0000110'0000 | State::HVM1, ENDSTATE}
        },

        // 10

        // pulse inversion
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(8e6f, 1, true, 2).toWaveform(),
            std::vector<uint32_t>{0b00000'0000110'0000 | State::HVM0, 0b00000'0000110'0000 | State::HVP0, ENDSTATE}
        },

        // 11

        // Custom waveform with:
        // 1-repetition
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
                {
                    WaveformSegment{
                        {2.0f/ SAMPLING_FREQUENCY, 4.0f/ SAMPLING_FREQUENCY, 7/ SAMPLING_FREQUENCY, 8/ SAMPLING_FREQUENCY, 34/ SAMPLING_FREQUENCY},
                        {-2, 2, 1, -1, 0}
                    },
                },
                {1}
            ),
            //                    2 cycles              4 cycles              7 cycles              8 cycles              34
            std::vector<uint32_t>{0b00000'0000000'0000 | State::HVM0, 0b00000'0000010'0000 | State::HVP0, 0b00000'0000101'0000 | State::HVP1, 0b00000'0000110'0000 | State::HVM1, 0b00000'0100000'0000 | State::CLAMP, ENDSTATE}
        },

        // 12

        // 2-state repetition
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
                {
                    WaveformSegment{
                        {8/ SAMPLING_FREQUENCY, 34/ SAMPLING_FREQUENCY},
                        {-2, 1}
                    },
                },
                {18}
                ),
            //                    8 cycles              34
            std::vector<uint32_t>{0b01000'0000110'0000 | State::HVM0, 0b00010'0100000'0000 | State::HVP1, ENDSTATE}
        },

        // 13

        // 3-state repetition
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
                {
                    WaveformSegment{
                        {8/ SAMPLING_FREQUENCY, 34/ SAMPLING_FREQUENCY, 7/ SAMPLING_FREQUENCY},
                        {-1, 2, 0}
                    },
                },
                {19}
                ),
            //                    8 cycles              34                     7
            std::vector<uint32_t>{0b10001'0000110'0000 | State::HVM1, 0b00010'0100000'0000 | State::HVP0,  0b00000'0000101'0000 | State::CLAMP, ENDSTATE}
        },

        // 14

        // 4-state repetition
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
                {
                    WaveformSegment{
                        {8/ SAMPLING_FREQUENCY, 34/ SAMPLING_FREQUENCY, 7/ SAMPLING_FREQUENCY, 2/ SAMPLING_FREQUENCY},
                        {-1, 2, 0, 1}
                    },
                },
                {((1 << 15) | (1 << 0) | (1 << 7)) + 2}
                ),
            //                    8 cycles              34                     7
            std::vector<uint32_t>{0b11001'0000110'0000 | State::HVM1, 0b10000'0100000'0000 | State::HVP0,  0b00000'0000101'0000 | State::CLAMP, 0b00100'0000000'0000 | State::HVP1, ENDSTATE}
        },

        // 15

        // Sequence of segments
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
                {
                    WaveformSegment{
                        {8/ SAMPLING_FREQUENCY, 34/ SAMPLING_FREQUENCY,},
                        {-1, 2}
                    },
                    WaveformSegment{
                        {7/ SAMPLING_FREQUENCY, 2/ SAMPLING_FREQUENCY},
                        {0, 1}
                    }
                },
                {1, 1}
                ),
            //                    8 cycles              34                     7                     2
            std::vector<uint32_t>{0b00000'0000110'0000 | State::HVM1, 0b00000'0100000'0000 | State::HVP0,  0b00000'0000101'0000 | State::CLAMP, 0b00000'0000000'0000 | State::HVP1, ENDSTATE}
        },

        // 16

        // single segment with long durations
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
                {
                    WaveformSegment{
                        {
                            0.5e-6, // less than max per register
                            129/130e6,  // exactly max
                            (129 + 1)/130e6,  // max + 1 cycle
                            (3*129 + 2)/130e6,  // max + 2 cycles
                            (7*129 + 1)/130e6, // 7*max + 1 max + rest
                            (8*129 + 5)/130e6, // 8*max + rest
                            (1033*129 + 3*129 + 1)/130e6
                        },
                        {
                            -1,
                            2,
                            0,
                            2,
                            1,
                            -2,
                            0
                        }
                    },
                },
                {1}
                ),
            std::vector<uint32_t>{
                // less than max
                0b00000'0111111'0000 | State::HVM1,
                // exactly max
                0b00000'1111111'0000 | State::HVP0,
                // max + 1 cycle => 128 cycles + 2 cycles
                0b00000'1111110'0000 | State::CLAMP,
                0b00000'0000000'0000 | State::CLAMP,
                // 3*max + 2 cycles
                0b00000'1111111'0000 | State::HVP0,
                0b00000'1111111'0000 | State::HVP0,
                0b00000'1111111'0000 | State::HVP0,
                0b00000'0000000'0000 | State::HVP0,
                // 7*max + 1 => 6*max + (max-1) + 2 => 3*(2*max) + (max-1) + 2
                0b01001'1111111'0000 | State::HVP1,
                0b00000'1111111'0000 | State::HVP1,
                0b00000'1111110'0000 | State::HVP1,
                0b00000'0000000'0000 | State::HVP1,
                // 8*max + 5 => 4*(2*max) + 5
                0b01010'1111111'0000 | State::HVM0,
                0b00000'1111111'0000 | State::HVM0,
                0b00000'0000011'0000 | State::HVM0,
                // 1033*max + 3 * max + 1 => 1032*max + 3 * max + (max-1) + 2 => 4*(256*max) + 3*max + (max-1) + 2
                0b11000'1111111'0000 | State::CLAMP,
                0b00000'1111111'0000 | State::CLAMP,
                0b00001'1111111'0000 | State::CLAMP,
                0b00000'1111111'0000 | State::CLAMP,
                // -
                0b00000'1111111'0000 | State::CLAMP,
                0b00000'1111111'0000 | State::CLAMP,
                0b00000'1111111'0000 | State::CLAMP,
                // -
                0b00000'1111110'0000 | State::CLAMP,
                // -
                0b00000'0000000'0000 | State::CLAMP,
                ENDSTATE
            }
        },

        // 17

        // "chirp-like" - 3 freqs
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Waveform(
            {
                    WaveformSegment{
                    {1/(2*560e3), 1/(2*560e3), 1/(2*320e3), 1/(2*320e3), 1/(2*80e3), 1/(2*80e3)},
                    {-2, 2, -2, 2, -2, 2}
                    }
                },
                {1}
            ),
            std::vector<uint32_t>{
                // 560 kHz, minus => < 1 us => 1 register = 116 cycles (114)
                0b00000'1110010'0000 | State::HVM0,
                // 560 kHz, plus => < 1 us => 1 register
                0b00000'1110010'0000 | State::HVP0,
                // 320 kHz, minus => 203 cycles = 129 + 74 cycles (72)
                0b00000'1111111'0000 | State::HVM0,
                0b00000'1001000'0000 | State::HVM0,
                // 320 kHz, plus
                0b00000'1111111'0000 | State::HVP0,
                0b00000'1001000'0000 | State::HVP0,
                // 80 kHz, minus => 813 cycles => 129 * 2 * 3 + 39 (37)
                0b01001'1111111'0000 | State::HVM0,
                0b00000'1111111'0000 | State::HVM0,
                0b00000'0100101'0000 | State::HVM0,
                // 80 kHz, plus
                0b01001'1111111'0000 | State::HVP0,
                0b00000'1111111'0000 | State::HVP0,
                0b00000'0100101'0000 | State::HVP0,
                ENDSTATE
            }
        },

        // 18

        // Other use cases
        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(0.5e6f, 2, false, 1).toWaveform(),
            ////             HVP0 (128 cycles) -> HVP0 (2 cycles) -> HVM0 (128 cycles) -> HVM0 (2 cycles). 4-state repetition
            std::vector<uint32_t>{0b11'000'1111110'0000 | State::HVP1, 0b00000'0000000'0000 | State::HVP1, 0b00000'1111110'0000 | State::HVM1, 0b00000'0000000'0000 | State::HVM1, ENDSTATE}
        },

        // 19

        TxWaveformConverterTestParams{
            // 0.5 MHz + 1 Tclk per level
            ::arrus::ops::us4r::Pulse(1.0/(2e-6 + 2.0/SAMPLING_FREQUENCY), 2, false, 1).toWaveform(),
            ////             HVP0 (129 cycles) -> HVP0 (2 cycles) -> HVM0 (129 cycles) -> HVM0 (2 cycles). 4-state repetition
            std::vector<uint32_t>{0b11'000'1111111'0000 | State::HVP1, 0b00000'0000000'0000 | State::HVP1, 0b00000'1111111'0000 | State::HVM1, 0b00000'0000000'0000 | State::HVM1, ENDSTATE}
        },

        // 20

        TxWaveformConverterTestParams{
            // 0.5 MHz + 1 Tclk per level
            ::arrus::ops::us4r::Pulse(1.0/(2e-6 + 4.0/SAMPLING_FREQUENCY), 2, false, 1).toWaveform(),
            ////             HVP0 (129 cycles) -> HVP0 (2 cycles) -> HVM0 (129 cycles) -> HVM0 (2 cycles). 4-state repetition
            std::vector<uint32_t>{0b11'000'1111111'0000 | State::HVP1, 0b00000'0000001'0000 | State::HVP1, 0b00000'1111111'0000 | State::HVM1, 0b00000'0000001'0000 | State::HVM1, ENDSTATE}
        },

        // 21

        TxWaveformConverterTestParams{
            ::arrus::ops::us4r::Pulse(0.5e6f, 4, false, 1).toWaveform(),
            ////             HVP0 (128 cycles) -> HVP0 (2 cycles) -> HVM0 (128 cycles) -> HVM0 (2 cycles). 4-state repetition
            std::vector<uint32_t>{0b11'010'1111110'0000 | State::HVP1, 0b00000'0000000'0000 | State::HVP1, 0b00000'1111110'0000 | State::HVM1, 0b00000'0000000'0000 | State::HVM1, ENDSTATE}
        }
    )
);
}
