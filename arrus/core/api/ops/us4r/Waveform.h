#ifndef ARRUS_CORE_API_OPS_US4R_WAVEFORM_H
#define ARRUS_CORE_API_OPS_US4R_WAVEFORM_H

#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

/**
 * Us4R TX waveform segment.
 *
 * @param time:
 */
class WaveformSegment {
public:
    WaveformSegment(const std::vector<float> &duration, const std::vector<int8_t> &state)
        : duration(duration), state(state) {}

    const std::vector<float> &getDuration() const { return duration; }
    const std::vector<int8_t> &getState() const { return state; }

    float getTotalDuration() const {
        float value = 0.0f;
        for(auto v: getDuration()) {
            value += v;
        }
        return value;
    }

private:
    std::vector<float> duration;
    std::vector<int8_t> state;
};

class Waveform {
public:
    Waveform(const std::vector<WaveformSegment> &segments, const std::vector<size_t> &nRepetitions)
        : segments(segments), nRepetitions(nRepetitions) {}

    const std::vector<WaveformSegment> &getSegments() const { return segments; }
    const std::vector<size_t> &getNRepetitions() const { return nRepetitions; }

    float getTotalDuration() const {
        float value = 0.0f;
        for(const auto &segment: getSegments()) {
            value += segment.getTotalDuration();
        }
        return value;
    }

private:
    std::vector<WaveformSegment> segments;
    std::vector<size_t> nRepetitions;
};

class WaveformBuilder {
public:

    WaveformBuilder& add(WaveformSegment segment, size_t nRepetitions = 1) {
        segments.emplace_back(std::move(segment));
        nReps.emplace_back(nRepetitions);
        return *this;
    }

    Waveform build() {
        return Waveform{segments, nReps};
    }

private:
    std::vector<WaveformSegment> segments;
    std::vector<size_t> nReps;
};

}

#endif//ARRUS_CORE_API_OPS_US4R_WAVEFORM_H
