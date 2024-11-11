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
    // Waveform sampling frequency
    constexpr static float SAMPLING_FREQUENCY = 130e6f;
    // Number of bits for the duration field
    constexpr static uint32_t DURATION_BITS_SIZE = 7;
    // MAXIMUM single duration; +1, because it's 127 + 2
    constexpr static uint32_t MAX_N_CYCLES = (1 << (DURATION_BITS_SIZE)) + 1;
    // MAX: 2^8-1+2
    constexpr static uint32_t MAX_2_STATE_REPTITIONS = (1 << 8) + 1;
    // MAX: 2^18-1+2, 18 = 3 + 5 + 5 + 5
    constexpr static uint32_t MAX_4_STATE_REPTITIONS = (1 << 18) + 1;
    // Maximum number of registers in a segment with repetition > 1.
    constexpr static uint32_t MAX_REPEATED_SEGMENT_LENGTH = 4;

    static void validateSegment(const ops::us4r::WaveformSegment &segment, const size_t nRepetitions) {
        auto segmentLength = segment.getState().size();
        if (segmentLength == 0) {
            throw IllegalArgumentException("Segment cannot be empty");
        }
        if (nRepetitions == 0) {
            throw IllegalArgumentException("The number of repetitions must be greater than 0");
        }

        if (nRepetitions > 1 && ((segmentLength < 2) || (segmentLength > 4))) {
            throw IllegalArgumentException("The repeated segment should have between 2 and 4 components.");
        }
        if (nRepetitions > 1) {
            const size_t maxNRepetitions = (1 << (3 + 5 * (segmentLength - 1))) + 2 - 1;
            if (nRepetitions > maxNRepetitions) {
                throw IllegalArgumentException(arrus::format(
                    "Exceeded maximum number of repetitions: '{}', value: '{}'", maxNRepetitions, nRepetitions));
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
        for (size_t s = 0; s < wf.getSegments().size(); ++s) {
            const auto &inputSegment = wf.getSegments().at(s);
            const auto nRepetitions = wf.getNRepetitions().at(s);
            std::vector<float> duration(inputSegment.getDuration().size());
            std::vector<int8_t> state(inputSegment.getState().size());
            for (size_t i = 0; i < inputSegment.getState().size(); ++i) {
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

    using SegmentWithRepetitions = std::pair<::arrus::ops::us4r::WaveformSegment, size_t>;

    static std::vector<SegmentWithRepetitions> handleTooManyRepetitions(const ::arrus::ops::us4r::WaveformSegment &segment, size_t nRepeats) {
        // Two rows with a two-element repetition: the maximum number of repetitions is 2^8+1 (8 bits + 2).
        // So, we should do the below conversions only for nRepeats > 257
        constexpr size_t minRepeats = 257;// 2-state repetitions
        if (nRepeats > minRepeats && segment.getState().size() == 2) {
            if (nRepeats % 2 == 0) {
                // Event number of repetitions.
                // In that case, the minimum number of rows we can use is 4.
                // (note: 3-state repetitions are not optimal -- we need to use 6-rows anyway get the proper states 3
                // and 4).
                std::vector<SegmentWithRepetitions> result;
                // Even number of repetitions: convert to 4-state repetitions
                result.emplace_back(
                    ::arrus::ops::us4r::WaveformSegment{{segment.getDuration().at(0), segment.getDuration().at(1),
                                                         segment.getDuration().at(0), segment.getDuration().at(1)},
                                                        {segment.getState().at(0), segment.getState().at(1),
                                                         segment.getState().at(0), segment.getState().at(1)}},
                    nRepeats / 2);
                return result;
            } else {
                // Odd number of repetitions => convert to 4 state repetitions.
                if (nRepeats < 2 * minRepeats) {
                    // The minimum number of entries is 4:
                    // 2-state even number repetitions + 2-state odd number of repetitions
                    std::vector<SegmentWithRepetitions> result;
                    result.emplace_back(
                        ::arrus::ops::us4r::WaveformSegment{{segment.getDuration().at(0), segment.getDuration().at(1)},
                                                            {segment.getState().at(0), segment.getState().at(1)}},
                        minRepeats);
                    // NOTE: nRepeats > minRepeats
                    // nRepeats is odd and minRepeats is odd => (nRepeats - minRepeats) is greater or equal than 2
                    result.emplace_back(
                        ::arrus::ops::us4r::WaveformSegment{{segment.getDuration().at(0), segment.getDuration().at(1)},
                                                            {segment.getState().at(0), segment.getState().at(1)}},
                        nRepeats - minRepeats);
                    return result;
                } else {
                    // The minimum number of entries is 6:
                    // 4-state repetitions + 2 states
                    std::vector<SegmentWithRepetitions> result;
                    // convert to 4-state repetitions (6 entries in total)
                    result.emplace_back(
                        ::arrus::ops::us4r::WaveformSegment{{segment.getDuration().at(0), segment.getDuration().at(1),
                                                             segment.getDuration().at(0), segment.getDuration().at(1)},
                                                            {segment.getState().at(0), segment.getState().at(1),
                                                             segment.getState().at(0), segment.getState().at(1)}},
                        nRepeats / 2);
                    // +1
                    result.emplace_back(
                        ::arrus::ops::us4r::WaveformSegment{{segment.getDuration().at(0), segment.getDuration().at(1)},
                                                            {segment.getState().at(0), segment.getState().at(1)}},
                        1// NOTE: "a single repetition" should be handled as "no repetition" later
                    );
                    return result;
                }
            }
        } else {
            // anything else: copy as it is
            // TODO: avoid copying?
            std::vector<SegmentWithRepetitions> result = {{segment, nRepeats}};
            return result;
        }
    }

    static std::vector<SegmentWithRepetitions> handleTooManyRepetitions(const std::vector<SegmentWithRepetitions> &segments) {
        std::vector<SegmentWithRepetitions> result;
        for(const auto &[s, n]: segments) {
            auto res = handleTooManyRepetitions(s, n);
            result.reserve(result.size() + std::distance(std::begin(res), std::end(res)));
            result.insert(std::end(result), std::begin(res), std::end(res));
        }
        return result;
    }

    static std::vector<SegmentWithRepetitions> preprocessSegment(const ::arrus::ops::us4r::WaveformSegment &segment,
                                                                 size_t nRepeats) {
        auto splittedSegments = handleTooLongStates(segment, nRepeats);
        return handleTooManyRepetitions(splittedSegments);
    }

    /**
     * Splits each state of the given segment into multiple shorter states in case a given state's duration is too
     * long to be handled by STHV1600 pulser.
     * The method splits the given segment into multiple smaller segments, if necessary. The output segments
     * guarantee, that there is no state with duration longer than the maximum duration for a single
     * pulser register.
     */
    static std::vector<SegmentWithRepetitions> handleTooLongStates(const ops::us4r::WaveformSegment &segment, size_t nRepeats) {
        std::vector<SegmentWithRepetitions> result;
        // Provide the possibility to transmit longer states (e.g., lower frequencies) by splitting and repeating
        // it if it's necessary.
        std::vector<float> currentDurations;
        std::vector<int8> currentStates;

        for (size_t i = 0; i < segment.getState().size(); ++i) {
            auto d = segment.getDuration().at(i);
            auto s = segment.getState().at(i);
            auto durationCycles = toClk(d);
            uint32_t reps = durationCycles / MAX_N_CYCLES;
            uint32_t rem1 = 0;
            uint32_t rem2 = durationCycles % MAX_N_CYCLES;
            // Make sure rest >= 2
            // NOTE: reps == 0 and rem2 == 1 is nto allowed for the pulser
            if (reps > 0 && rem2 == 1) {
                // This in order to guarantee that rem2 is always >= 2
                rem2 = 2;               // increase by 1
                rem1 = MAX_N_CYCLES - 1;// Move one rep to rem1, and subtract 1 (from the rem2)
                reps -= 1;              // one rep has been moved to the rem1
            }
            auto nStateEntries = reps + std::clamp<uint32_t>(rem1, 0, 1) + std::clamp<uint32_t>(rem2, 0, 1);
            if (rem1 == 0 && reps == 0) {
                // duration short enough to be stored in a single waveform register
                // Just copy the current segment as it is.
                currentDurations.push_back(d);
                currentStates.push_back(s);
            } else if (nStateEntries <= 4 || nRepeats > 1) {
                // - If the given state duration def requires up to 4 entries (the max number of
                // - If the number of waveform repeats > 1, we cannot use the pulser 2/3/4-state repetitions.
                // NOTE: the given number of repetitions may still not be achievable due to the total number of states
                // which must be less or equal 4.
                // !! NOTE: the value "4" is a heuristically selected value -- it may not be the optimal.
                // The value 5 = 4 + 1 corresponds to the worst case for the reps/2 < MAX_2_STATE_REPETITIONS scenario.
                // The value might be adjusted in the future e.g. based on the nRepeats and rem1, rem2 values.
                for (uint32_t j = 0; j < reps; ++j) {
                    currentDurations.push_back(toTime(MAX_N_CYCLES));
                    currentStates.push_back(s);
                }
                if (rem1 > 0) {
                    currentDurations.push_back(toTime(rem1));
                    currentStates.push_back(s);
                }
                if (rem2 > 0) {
                    currentDurations.push_back(toTime(rem2));
                    currentStates.push_back(s);
                }
            } else {
                // reps >= 4 => try via 2-element segment repetitions
                if (!currentDurations.empty()) {
                    // Close the currently visited elements as a single segment.
                    result.emplace_back(ops::us4r::WaveformSegment(currentDurations, currentStates), size_t(1));
                    currentDurations.clear();
                    currentStates.clear();
                }
                if (reps / 2 <= MAX_2_STATE_REPTITIONS) {
                    // 2-state repetition
                    size_t segmentNReps = reps / 2;
                    size_t repsRem = reps % 2;
                    result.emplace_back(
                        ops::us4r::WaveformSegment({toTime(MAX_N_CYCLES), toTime(MAX_N_CYCLES)}, {s, s}), segmentNReps);
                    if (repsRem > 0) {
                        // repsRem == 1
                        result.emplace_back(ops::us4r::WaveformSegment({toTime(MAX_N_CYCLES)}, {s}), 1);
                    }
                    if (rem1 > 0) {
                        result.emplace_back(ops::us4r::WaveformSegment({toTime(rem1)}, {s}), 1);
                    }
                    if (rem2 > 0) {
                        result.emplace_back(ops::us4r::WaveformSegment({toTime(rem2)}, {s}), 1);
                    }
                } else if (reps / 4 <= MAX_4_STATE_REPTITIONS) {
                    // 4-state repetition
                    size_t segmentNReps = reps / 4;
                    size_t repsRem = reps % 4;
                    result.emplace_back(ops::us4r::WaveformSegment({toTime(MAX_N_CYCLES), toTime(MAX_N_CYCLES),
                                                                    toTime(MAX_N_CYCLES), toTime(MAX_N_CYCLES)},
                                                                   {s, s, s, s}),
                                        segmentNReps);
                    if (repsRem > 0) {
                        std::vector<float> ds(repsRem, toTime(MAX_N_CYCLES));
                        std::vector<int8> ss(repsRem, s);
                        result.emplace_back(ops::us4r::WaveformSegment(ds, ss), 1);
                    }
                    if (rem1 > 0) {
                        result.emplace_back(ops::us4r::WaveformSegment({toTime(rem1)}, {s}), 1);
                    }
                    if (rem2 > 0) {
                        result.emplace_back(ops::us4r::WaveformSegment({toTime(rem2)}, {s}), 1);
                    }
                } else {
                    throw IllegalArgumentException(
                        format("The TX waveform includes too long single state duration: {}", d));
                }
            }
        }
        // clean up anything was currently created.
        if (!currentDurations.empty()) {
            // Close the currently visited elements as a single segment.
            result.emplace_back(ops::us4r::WaveformSegment(currentDurations, currentStates), 1);
            currentDurations.clear();
            currentStates.clear();
        }
        if(nRepeats > 1) {
            // Try to make a single segment with a given number of repetitions.
            // But also check, if the following condition is satisfied: the total number of states in all segments
            // is no greater than 4.
            std::vector<int8> repeatedSegmentStates;
            std::vector<float> repeatedSegmentDurations;
            for(const auto &[subSegment, n]: result) {
                for(uint32_t i = 0; i < n; ++i) {
                    for(size_t j = 0; j < subSegment.getState().size(); ++j) {
                        const auto &subState = subSegment.getState().at(j);
                        const auto &subStateDuration = subSegment.getDuration().at(j);
                        repeatedSegmentStates.push_back(subState);
                        repeatedSegmentDurations.push_back(subStateDuration);
                    }
                }
            }
            if(repeatedSegmentStates.size() > MAX_REPEATED_SEGMENT_LENGTH) {
                throw IllegalArgumentException(
                    format(
                        "The TX waveform has too long single state duration(s) to be repeated the given "
                        "number of times '{}' ", nRepeats));
            }
            // Just a single waveform with a given number of repeats.
            return {{
                ::arrus::ops::us4r::WaveformSegment{repeatedSegmentDurations, repeatedSegmentStates},
                nRepeats
            }};
        }
        else {
            return result;
        }
    }

    static std::vector<uint32_t> toPulser(const ::arrus::ops::us4r::Waveform &wf) {
        std::vector<uint32_t> result;
        std::vector<SegmentWithRepetitions> preprocessedSegments;
        for (size_t s = 0; s < wf.getSegments().size(); ++s) {
            // TODO avoid copying here?
            const auto &segment = wf.getSegments().at(s);
            const auto nRepetitions = ARRUS_SAFE_CAST(wf.getNRepetitions().at(s), size_t);
            const auto preprocessed = preprocessSegment(segment, nRepetitions);
            preprocessedSegments.insert(std::end(preprocessedSegments), std::begin(preprocessed),
                                        std::end(preprocessed));
        }

        for (const auto &[segment, nRepetitions] : preprocessedSegments) {
            validateSegment(segment, nRepetitions);

            for (size_t i = 0; i < segment.getState().size(); ++i) {
                uint32_t reg = 0;

                auto duration = segment.getDuration().at(i);
                if (duration <= 0.0f) {
                    throw IllegalArgumentException("The waveform duration time should be non-negative");
                }
                auto state = segment.getState().at(i);

                uint32 durationClk = toClk(duration);
                if (durationClk < 2) {
                    throw IllegalArgumentException(format(
                        "The minimum number of cycles that can be set on the pulser is 2, got: {}", durationClk));
                }
                // The actual number of cycles is durationClk + 2 cycles
                durationClk -= 2;
                auto pulserState = getDeviceState(state);
                reg = setState(reg, pulserState);
                reg = setDuration(reg, durationClk);
                if (nRepetitions == 1) {
                    reg = setNRepetitionsPart(reg, 0, 0);
                } else {
                    // nRepetitions >= 2
                    auto actualNRepetitions = ARRUS_SAFE_CAST(nRepetitions - 2, uint32_t);
                    auto repetitionType = getRepetitionType(segment);
                    if (i == 0) {
                        reg = setRepeatType(reg, repetitionType);
                        reg = setNRepetitionsPart(reg, actualNRepetitions, ARRUS_SAFE_CAST(i, uint32_t));
                    } else {
                        reg = setNRepetitionsPart(reg, actualNRepetitions, ARRUS_SAFE_CAST(i, uint32_t));
                    }
                }

                result.push_back(reg);
            }
        }
        result.push_back(END_STATE);
        if (result.size() > 256) {
            throw IllegalArgumentException("Exceeded maximum pulser waveform memory.");
        }
        return result;
    }

private:
    static uint32_t toClk(float time) { return static_cast<uint32>(std::roundf(time * SAMPLING_FREQUENCY)); }

    static float toTime(uint32_t clk) { return static_cast<float>(clk) / SAMPLING_FREQUENCY; }

    static uint32_t setNRepetitionsPart(uint32_t input, uint32_t nRepetitions, uint32_t part) {
        uint32_t mask = 0;
        uint32_t size = 0;
        uint32_t offset = 0;
        if (part == 0) {
            offset = 0;
            mask = 0b111;
            size = 3;
        } else {
            offset = (3 + 5 * (part - 1));
            mask = 0b11111 << offset;
            size = 5;
        }
        return setBitField(input, 11, size, (nRepetitions & mask) >> offset);
    }

    static uint32_t getDeviceState(int8 apiState) {
        switch (apiState) {
        case -1:
            // HVM0
            return 0b1010;
        case 1:
            // HVP0
            return 0b0101;
        case -2:
            //HVM1
            return 0b1001;
        case 2:
            // HVP1
            return 0b0110;
        case 0:
            // CLAMP
            return 0b1111;
        default: throw IllegalArgumentException(format("Unrecognized waveform state: {}", apiState));
        }
    }

    static uint32_t getRepetitionType(const ::arrus::ops::us4r::WaveformSegment &segment) {
        auto repetitionType = ARRUS_SAFE_CAST(segment.getState().size(), uint32_t);
        if (repetitionType == 0 || repetitionType > 4) {
            throw IllegalArgumentException(
                "The repeated waveform segment must have between 1 and 4 components (inclusive).");
        }
        return repetitionType - 1;
    }

    static uint32_t setState(uint32_t input, uint32_t value) {
        constexpr size_t size = 4;
        size_t maxValue = (1 << size);
        if (value >= maxValue) {
            throw IllegalArgumentException(format("Waveform state '{}' should be less than '{}'", value, maxValue));
        }
        return setBitField(input, 0, size, value);
    }

    static uint32_t setDuration(uint32_t input, uint32_t value) {
        constexpr size_t size = DURATION_BITS_SIZE;
        size_t maxValue = (1 << size);
        if (value >= maxValue) {
            throw IllegalArgumentException(
                format("Waveform number of cycles '{}' should be less than '{}'", value, maxValue));
        }
        return setBitField(input, 4, size, value);
    }

    static uint32_t setRepeatType(uint32_t input, uint32_t type) {
        constexpr size_t size = 2;
        size_t maxValue = (1 << size);
        if (type >= maxValue) {
            throw IllegalArgumentException(
                format("Waveform repeat type '{}' should be less than '{}'", type, maxValue));
        }
        return setBitField(input, 14, size, type);
    }

    static uint32_t setBitField(uint32_t input, uint32_t offset, uint32_t size, uint32_t value) {
        auto mask = static_cast<uint32_t>(((1 << size) - 1) << offset);
        return ((input & ~mask) | ((value << offset) & mask));
    }
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_TXWAVEFORMCONVERTER_H
