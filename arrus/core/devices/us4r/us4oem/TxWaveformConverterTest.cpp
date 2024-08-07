#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include "TxWaveformConverter.h"

namespace {
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;

struct BuildSequenceWaveformParams {
    Waveform input;
    std::vector<uint32_t> expectedOutput;
};

class Sequencer2BuildsCorrectWaveformTest : public Sequencer2Test<BuildSequenceWaveformParams> {};

TEST_P(Sequencer2BuildsCorrectWaveformTest, BuildsCorrectWaveform) {
    // Given
    Sequencer2 sequencer2(pciDevice, TEST_BAR, (BaseAddress) ((size_t) registers.get()), mockTxPulsers);
    std::bitset<128/8> hizen(0xFFFF);
    std::bitset<128> rxen;
    std::bitset<128> txen;
    auto freq = GetParam().setFrequency;
    auto nop = GetParam().setNop;
    auto firings = GetParam().nFirings;
    auto freqStep = GetParam().freqStep;

    auto expectedLength = GetParam().expectedWaveformLength;
    auto expectedFirstWord = GetParam().expectedFirstByte;

    for(uint8_t i = 0; i<128; i++) {
        txen.set(i, true);
        rxen.set(i, true);
    }

    sequencer2.SetNumberOfFirings(firings);

    for(uint8_t i = 0; i < firings; i++) {
        sequencer2.SetRxEn(rxen, i);
        sequencer2.SetTxEn(txen, i);
        sequencer2.SetRxDelay(0.0f, i);
        sequencer2.SetDelay(0, 0.0f, i, 0);
        sequencer2.SetHiZEn(hizen, i);
        sequencer2.SetOCWSFrequency(freq+(freqStep*i), i);
        sequencer2.SetOCWSNop(nop, i);
        sequencer2.SetVoltageLevel(0, i);
    }

    for(uint8_t i = 0; i < 8; i++) {
        for(uint8_t j = 0; j < expectedFirstWord.size(); j++) {
            EXPECT_CALL(*(mockTxPulsersRaw[i]), WriteWaveform(_, _, CArrayFirstEqual(expectedFirstWord[j]), expectedLength)).Times(1);
        }
    }

    for(uint8_t i = 0; i < firings; i++) { sequencer2.BuildSequenceWaveform(i); }
}

INSTANTIATE_TEST_CASE_P(
    CorrectWaveformParams, Sequencer2BuildsCorrectWaveformTest,
    testing::Values(
        BuildSequenceWaveformParams{1e6f, 2, 1, 0.0f, 3, std::vector<uint16_t>{0x03F5}}, //1 firing, 1 MHz, 2 states, no repetition
        BuildSequenceWaveformParams{8e6f, 2, 8, 0.0f, 3, std::vector<uint16_t>{0x0065}}, //8 firings, 8 MHz, 2 states (same waveform so expect 1 write), no repetition
        BuildSequenceWaveformParams{1e6f, 8, 1, 0.0f, 3, std::vector<uint16_t>{0x53F5}}, //1 firing, 1 MHz, 8 states, 2-state repetition
        BuildSequenceWaveformParams{8e6f, 8, 8, 0.0f, 3, std::vector<uint16_t>{0x5065}}, //8 firings, 8 MHz, 8 states (same waveform so expect 1 write), 2-state repetition
        BuildSequenceWaveformParams{1e6f, 1, 1, 0.0f, 2, std::vector<uint16_t>{0x03F5}}, //1 firing, 1 MHz, 1 state
        BuildSequenceWaveformParams{1e6f, 3, 1, 0.0f, 4, std::vector<uint16_t>{0x7BF5}}, //1 firing, 1 MHz, 3 state
        BuildSequenceWaveformParams{1e6f, 2, 8, 1e6f, 3, std::vector<uint16_t>{0x03F5, 0x01F5, 0x0145, 0xE5, 0xB5, 0x95, 0x75, 0x65}}, //8 firings, 1 - 8 MHz (1 MHz step), 2 states (different waveforms so expect 8 writes)
        BuildSequenceWaveformParams{1e6f, 1024, 1, 0.0f, 17, std::vector<uint16_t>{0xC8E5}})); //1 firing, 1 MHz, 1024 states (soft start enabled = 17 states, C8E5 = REP4STATES, + repeat + time + HVP1 pulse)

}