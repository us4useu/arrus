#ifndef ARRUS_CORE_DEVICES_US4R_TXWAVEFORMCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_TXWAVEFORMCONVERTER_H

#include "arrus/common/format.h"
#include "arrus/common/utils.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/ops/us4r/Waveform.h"
#include <vector>

namespace arrus::devices {

/**
 * Converts tx waveform from the HAL waveform description to the STHV 1600 pulser memory definition.
 */
class TxWaveformConverter {
public:
    constexpr static uint32_t END_STATE = 0b1110;

    static void validateSegment(const ops::us4r::WaveformSegment &segment, const uint32_t nRepetitions) {
        auto segmentLength = segment.getState().size();
        if(segmentLength == 0) {
            throw IllegalArgumentException("Segment cannot be empty");
        }
        if(nRepetitions == 0) {
            throw IllegalArgumentException("The number of repetitions must be greater than 0");
        }

        if(nRepetitions > 1 && ((segmentLength < 2) || (segmentLength > 4))) {
            throw IllegalArgumentException("The repeated segment should have between 2 and 4 components.");
        }
        if(nRepetitions > 1) {
            const uint32_t maxNRepetitions = (1 << (3 + 5*(segmentLength-1))) + 2 - 1;
            if(nRepetitions >= maxNRepetitions) {
                throw IllegalArgumentException(
                    arrus::format("Exceeded maximum number of repetitions: '{}', value: '{}'", maxNRepetitions, nRepetitions));
            }
        }
    }

    /**
     * Converts the input waveform to a waveform that will be actually applied on the device.
     * In particular, applies the actual waveform sampling frequency.
     *
     * @param wf input waveform
     * @return waveform after considering the device sampling frequency
     */
    static ::arrus::ops::us4r::Waveform getHWWaveform(const ::arrus::ops::us4r::Waveform &wf) {
        ops::us4r::WaveformBuilder builder;
        for(size_t s = 0; s < wf.getSegments().size(); ++s) {
            const auto &inputSegment = wf.getSegments().at(s);
            const auto nRepetitions = wf.getNRepetitions().at(s);
            std::vector<float> duration(inputSegment.getDuration().size());
            std::vector<int8_t> state(inputSegment.getState().size());
            for(size_t i = 0; i < inputSegment.getState().size(); ++i) {
                const auto inputDuration = inputSegment.getDuration().at(i);
                float actualDuration = toTime(toClk(inputDuration));
                // Set
                duration.at(i) = actualDuration;
                state.at(i) = inputSegment.getState().at(i);
            }
            builder.add(ops::us4r::WaveformSegment(duration, state), nRepetitions);
        }
        return builder.build();
    }

    static std::vector<uint32_t> toPulser(const ::arrus::ops::us4r::Waveform &wf) {
        std::vector<uint32_t> result;
        for(size_t s = 0; s < wf.getSegments().size(); ++s) {
            const auto &segment = wf.getSegments().at(s);
            auto nRepetitions = ARRUS_SAFE_CAST(wf.getNRepetitions().at(s), uint32_t);

            validateSegment(segment, nRepetitions);

            auto repetitionType = getRepetitionType(segment);
            for(size_t i = 0; i < segment.getState().size(); ++i) {
                uint32_t reg = 0;

                auto duration = segment.getDuration().at(i);
                if(duration <= 0.0f) {
                    throw IllegalArgumentException("The waveform duration time should be non-negative");
                }
                auto state = segment.getState().at(i);

                uint32 durationClk = toClk(duration);
                if(durationClk < 2) {
                    throw IllegalArgumentException(
                        format("The minimum number of cycles that can be set on the pulser is 2, got: {}", durationClk));
                }
                // The actual number of cycles is durationClk + 2 cycles
                durationClk -= 2;
                auto pulserState = getDeviceState(state);
                reg = setState(reg, pulserState);
                reg = setDuration(reg, durationClk);
                if(nRepetitions == 1) {
                    reg = setNRepetitionsPart(reg, 0, 0);
                }
                else {
                    // nRepetitions >= 2
                    uint32_t actualNRepetitions = nRepetitions - 2;
                    if(i == 0) {
                        reg = setRepeatType(reg, repetitionType);
                        reg = setNRepetitionsPart(reg, actualNRepetitions, i);
                    }
                    else {
                        reg = setNRepetitionsPart(reg, actualNRepetitions, i);
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
    constexpr static float SAMPLING_FREQUENCY = 130e6f;

    static uint32_t toClk(float time) {
        return static_cast<uint32>(std::roundf(time*SAMPLING_FREQUENCY));
    }

    static float toTime(uint32_t clk) {
        return static_cast<float>(clk)/SAMPLING_FREQUENCY;
    }

    static uint32_t setNRepetitionsPart(uint32_t input, uint32_t nRepetitions, size_t part) {
        uint32_t mask = 0;
        uint32_t size = 0;
        if(part == 0) {
            mask = 0b111;
            size = 3;
        }
        else {
            mask = 0b11111 << (3 + 5*(part-1));
            size = 5;
        }
        return setBitField(input, 11, size, nRepetitions & mask);
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
        return setBitField(input, 0, size, value);
    }

    static uint32_t setDuration(uint32_t input, uint32_t value) {
        constexpr size_t size = 7;
        size_t maxValue = (1 << size);
        if(value >= maxValue) {
            throw IllegalArgumentException(format("Waveform number of cycles '{}' should be less than '{}'", value, maxValue));
        }
        return setBitField(input, 4, size, value);
    }

    static uint32_t setRepeatType(uint32_t input, uint32_t type) {
        constexpr size_t size = 2;
        size_t maxValue = (1 << size);
        if(type >= maxValue) {
            throw IllegalArgumentException(format("Waveform repeat type '{}' should be less than '{}'", type, maxValue));
        }
        return setBitField(input, 14, size, type);
    }

    static uint32_t setBitField(uint32_t input, uint32_t offset, uint32_t size, uint32_t value) {
        auto mask = static_cast<uint32_t>(((1 << size)-1) << offset);
        return ((input & ~mask) | ((value << offset) & mask));
    }
};

}

#endif//ARRUS_CORE_DEVICES_US4R_TXWAVEFORMCONVERTER_H
