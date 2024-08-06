#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_TXWAVEFORMCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_TXWAVEFORMCONVERTER_H

#include "arrus/core/api/ops/us4r/Waveform.h"
#include <vector>

namespace arrus::devices {

/**
 * Converts tx waveform from the HAL waveform description to the STHV 1600 pulser memory definition.
 */
class TxWaveformConverter {
public:
    static void validateSegment(const ops::us4r::WaveformSegment &segment, const uint32_t nRepetitions) {
        auto segmentLength = segment.getState().size();
        if(segmentLength == 0) {
            throw IllegalArgumentException("Segment cannot be empty");
        }
        if(nRepetitions == 0) {
            throw IllegalArgumentException("The number of repetitions must be greater than 0");
        }

        if(nRepetitions > 1 && (segmentLength < 2 | segmentLength > 4)) {
            throw IllegalArgumentException("The repeated segment should have between 2 and 4 components.");
        }
        if(nRepetitions > 1) {
            const uint32_t maxNRepetitions = (1 << (3 + 5*(segmentLength-1))) + 2 - 1;
            if(nRepetitions >= maxNRepetitions) {
                throw IllegalArgumentException(
                    format("Exceeded maximum number of repetitions: '{}', value: '{}'", maxNRepetitions, nRepetitions));
            }
        }
    }

    static std::vector<uint32_t> toPulser(const ::arrus::ops::us4r::Waveform &wf) {
        std::vector<uint32_t> result;
        for(size_t s = 0; s < wf.getSegments().size(); ++s) {
            const auto &segment = wf.getSegments().at(s);
            auto nRepetitions = ARRUS_SAFE_CAST(wf.getNRepetitions().at(s), uint32_t);

            validateSegment(segment, nRepetitions);

            auto repetitionType = getRepetitionType(segment);
            for(size_t x = 0; x < segment.getState().size(); ++x) {
                uint32_t reg = 0;

                auto duration = segment.getDuration().at(x);
                if(duration <= 0.0f) {
                    throw IllegalArgumentException("The waveform duration time should be non-negative");
                }
                auto state = segment.getState().at(x);

                uint32 durationClk = static_cast<uint32>(std::roundf(duration*SAMPLING_FREQUENCY));
                if(durationClk < 2) {
                    throw IllegalArgumentException(
                        format("The minimum number of cycles that can be set on the pulser is 2, got: {}", durationClk));
                }
                // The actual number of cycles is durationClk + 2 cycles
                durationClk -= 2;
                auto pulserState = getDeviceState(state);
                reg = setState(reg, pulserState);
                reg = setDuration(reg, durationClk);
                if(repetitionType == 0) {
                    // nRepetitions == 1
                    reg = setNRepetitionsPart(reg, 0, 0);
                }
                else {
                    // nRepetitions == 2 or more
                    nRepetitions -= 2;
                    if(x == 0) {
                        reg = setRepeatType(reg, repetitionType);
                        reg = setNRepetitionsPart(reg, nRepetitions, x);
                    }
                    else {
                        reg = setNRepetitionsPart(reg, nRepetitions, x);
                    }
                }

                result.push_back(reg);
            }
        }
        result.push_back(END_STATE);
        if(result.size() > 256) {
            throw IllegalArgumentException("Exceeded maximum pulser waveform memory.");
        }
        return result;
    }
private:
    constexpr static float SAMPLING_FREQUENCY = 130e6f / 2.0f;
    constexpr static uint32_t END_STATE = 0b1110;

    static uint32_t setNRepetitionsPart(uint32_t input, uint32_t nRepetitions, size_t part) {
        uint32_t mask = 0;
        if(part == 0) {
            mask = 0b111;
        }
        else {
            mask = 0b11111 << (3 + 5*(part-1));
        }
        return setBitField(input, 11, nRepetitions & mask);
    }

    static uint32_t getDeviceState(int8 apiState) {
        switch(apiState) {
            case -1:
                // HVM0
                return 0b1010;
                // HVP0
            case 1:
                return 0b0101;
            case 2:
                //HVM1
                return 0b1001;
            case -2:
                // HVP1
                return 0b0110;
            case 0:
                // CLAMP
                return 0b1111;
            default:
                throw IllegalArgumentException(format("Unrecognized waveform state: {}", apiState));
            }
    }

    static uint32_t getRepetitionType(const ::arrus::ops::us4r::WaveformSegment &segment) {
        auto repetitionType = segment.getState().size();
        if(repetitionType == 0 || repetitionType > 4) {
            throw IllegalArgumentException("The repeated waveform segment must have between 1 and 4 components (inclusive).");
        }
        return repetitionType-1;
    }

    static uint32_t setState(uint32_t input, uint32_t value) {
        constexpr size_t size = 4;
        size_t maxValue = (1 << size);
        if(value >= maxValue) {
            throw IllegalArgumentException(format("Waveform state '{}' should be less than '{}'", value, maxValue));
        }
        return setBitField(input, 0, value);
    }

    static uint32_t setDuration(uint32_t input, uint32_t value) {
        constexpr size_t size = 7;
        size_t maxValue = (1 << size);
        if(value >= maxValue) {
            throw IllegalArgumentException(format("Waveform number of cycles '{}' should be less than '{}'", value, maxValue));
        }
        return setBitField(input, 4, value);
    }

    static uint32_t setRepeatType(uint32_t input, uint32_t type) {
        constexpr size_t size = 2;
        size_t maxValue = (1 << size);
        if(type >= maxValue) {
            throw IllegalArgumentException(format("Waveform repeat type '{}' should be less than '{}'", type, maxValue));
        }
        return setBitField(input, 14, type);
    }

    static uint32_t setBitField(uint32_t input, uint32_t offset, uint32_t value) {
        value = value << offset;
        return (input & value) | value;
    }
};

}

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_TXWAVEFORMCONVERTER_H
