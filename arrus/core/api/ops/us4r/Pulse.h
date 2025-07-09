#ifndef ARRUS_CORE_API_OPS_US4R_PULSE_H
#define ARRUS_CORE_API_OPS_US4R_PULSE_H

#include <algorithm>
#include <cmath>
#include <ostream>

#include "Waveform.h"

namespace arrus::ops::us4r {

/**
 * A single pulse (sine wave) produced by us4r device.
 *
 * DEPRECATED: please use the arrus::ops::us4r::Waveform
 */
class Pulse {
public:
    using AmplitudeLevel = uint8;

    /**
	 * Pulse constructor.
	 *
	 * @param centerFrequency center frequency of the transmitted pulse
	 * @param nPeriods pulse number of periods, should be a multiple of 0.5
	 * @param inverse if set to true - inverse the pulse polarity
	 * @param amplitudeLevel amplitude level to use, default: 2 (HVM/P 0)
	 */
    Pulse(float centerFrequency, float nPeriods, bool inverse, AmplitudeLevel amplitudeLevel = 2)
        : centerFrequency(centerFrequency), nPeriods(nPeriods), inverse(inverse), amplitudeLevel(amplitudeLevel) {
        if(! (amplitudeLevel == 1 || amplitudeLevel == 2)) {
            throw IllegalArgumentException("Pulse amplitude level should be 1 or 2");
        }
    }

    float getCenterFrequency() const {
        return centerFrequency;
    }

    float getNPeriods() const {
        return nPeriods;
    }

    bool isInverse() const { return inverse; }

    AmplitudeLevel getAmplitudeLevel() const { return amplitudeLevel; }

    bool operator==(const Pulse &rhs) const {
        return centerFrequency == rhs.centerFrequency
               && nPeriods == rhs.nPeriods
               && inverse == rhs.inverse
               && amplitudeLevel == rhs.amplitudeLevel;
    }

    bool operator!=(const Pulse &rhs) const {
        return !(rhs == *this);
    }

    /**
     * Returns pulse duration [s].
     */
    float getPulseLength() const {
        return nPeriods/centerFrequency;
    }

    Waveform toWaveform() const {
        WaveformBuilder wb;
        float t = 1.0f/getCenterFrequency();
        int8 polarity = isInverse() ? -1: 1;

        auto currentNPeriods = getNPeriods();

        if(currentNPeriods >= 1.0f) {
            WaveformSegment segment{
                {t/2, t/2},
                {(int8)(polarity*amplitudeLevel), (int8)(-1*polarity*amplitudeLevel)}
            };
            float integral, fractional;
            fractional = std::modf(nPeriods, &integral);
            auto nRepetitions = static_cast<size_t>(integral);
            wb.add(segment, nRepetitions);
            currentNPeriods = fractional;
        }
        if(currentNPeriods > 0.0f) {
            auto rest = std::clamp(currentNPeriods, 0.0f, 0.5f);
            WaveformSegment rem1{
                {rest *t},
                {(int8)(polarity*amplitudeLevel)}
            };
            wb.add(rem1);
            rest = currentNPeriods- rest;
            if(rest > 0.0f) {
                WaveformSegment rem2{
                    {rest *t},
                    {(int8)(-1*polarity*amplitudeLevel)}
                };
                wb.add(rem2);
            }
        }
        return wb.build();
    }

    /**
     * Converts the input waveform to a pulse.
     * The conversion is possible only when the structure of the waveform is conformant with the output
     * of the toWaveform method -- the nullopt is returned otherwise.
     *
     * @return pulse recovered from the input waveform or std::nullopt, if it was not possible to convert the
     * waveform to the Pulse object
     */
    static std::optional<Pulse> fromWaveform(const Waveform& waveform) {
        if(waveform.getSegments().empty() || waveform.getSegments().size() > 3) {
            return std::nullopt;
        }
        const auto &seg0 = waveform.getSegments().at(0);
        float period = 0.0f;
        bool isInverse = false;
        float nCycles = 0.0f;
        uint8 level = 0;

        if(seg0.getState().size() > 2) {
            // We expect at most two states.
            return std::nullopt;
        }

        if(seg0.getState().size() == 2) {
            // two states -- at least a single full period.
            if(seg0.getState().at(0) != -1*seg0.getState().at(1)) {
                // amplitude -> -amplitude sequence is expected
                return std::nullopt;
            }
            auto signedState = seg0.getState().at(0);
            auto state = abs(signedState);
            if(state != 1 && state != 2) {
                // Unacceptable state in the beginning of the waveform.
                return std::nullopt;
            }
            if(seg0.getDuration().at(0) != seg0.getDuration().at(1)) {
                // t/2 -> t/2 is expected
                return std::nullopt;
            }
            level = static_cast<uint8>(state);
            period = seg0.getDuration().at(0) * 2.0f;
            nCycles = static_cast<float>(waveform.getNRepetitions().at(0));
            isInverse = seg0.getState().at(0) < 0;

            // If we have more segments -- make sure the rest is compatible with the current parameters, and update
            // ncycles appropriately
            if(waveform.getSegments().size() >= 2) {
                // full cycle, 0.5, rest
                const auto &seg1 = waveform.getSegments().at(1);
                auto t1Update = getNCyclesFractional(seg1, signedState, period);
                if(!t1Update.has_value()) {
                    return std::nullopt;
                }
                else {
                    // NOTE: we are ignoring here the situation where t < 0.5!
                    nCycles += t1Update.value();
                }

                if(waveform.getSegments().size() == 3) {
                    const auto &seg2 = waveform.getSegments().at(2);
                    auto t2Update = getNCyclesFractional(seg2, -signedState, period);
                    if(!t2Update.has_value()) {
                        return std::nullopt;
                    }
                    else {
                        nCycles += t2Update.value();
                    }
                }
            }
        }
        else {
            if(waveform.getSegments().size() > 2) {
                // We expect only two segments (up and down)
                return std::nullopt;
            }
            auto signedState = seg0.getState().at(0);
            auto state = abs(signedState);
            if(state != 1 && state != 2) {
                // Unacceptable state in the beginning of the waveform.
                return std::nullopt;
            }
            level = static_cast<uint8>(state);
            isInverse = seg0.getState().at(0) < 0;
            period = seg0.getDuration().at(0) * 2.0f;
            nCycles = 0.5f;

            if(waveform.getSegments().size() == 2) {
                const auto &seg1 = waveform.getSegments().at(1);
                if(seg1.getState().size() != 1) {
                    // A single state is expected
                    return std::nullopt;
                }
                if(seg1.getState().at(0) != -signedState) {
                    // Opposite state is expected.
                    return std::nullopt;
                }
                if(seg1.getDuration().at(0) > seg0.getDuration().at(0)) {
                    // The second state should not be longer that the first state.
                    return std::nullopt;
                }
                nCycles += seg1.getDuration().at(0)/seg0.getDuration().at(0)*0.5f;
            }
        }
        return Pulse{
            1/period,
            nCycles,
            isInverse,
            level
        };
    }

    friend std::ostream &operator<<(std::ostream &os, const Pulse &pulse) {
        os << "centerFrequency: " << pulse.centerFrequency << " nPeriods: " << pulse.nPeriods
           << " inverse: " << pulse.inverse << " amplitudeLevel: " << pulse.amplitudeLevel;
        return os;
    }

private:
    static bool areAlmostEqual(float a, float b, float atol) {
        return std::fabs(a - b) < atol;
    }

    static std::optional<float> getNCyclesFractional(
        const WaveformSegment &segment, int8 expectedState, float period) {
        if(segment.getDuration().size() > 1) {
            // Each segment should contain a single state
            return std::nullopt;
        }
        auto t = segment.getDuration().at(0);
        auto s = segment.getState().at(0);
        if(s != expectedState) {
            // s1 and s2 should be the same as the states of the first segments
            return std::nullopt;
        }
        if(t > 0.5*period) {
            // should not be higher than 0.5 T
            return std::nullopt;
        }
        if(areAlmostEqual(t, 0.5, 1e-2f)) {
            // Set exactly 0.5 -- this is value acceptable by the legacy OEMs.
            return 0.5f;
        }
        else {
            return t/period;
        }
    }

    float centerFrequency;
    float nPeriods;
    bool inverse;
    AmplitudeLevel amplitudeLevel = 2;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_PULSE_H
