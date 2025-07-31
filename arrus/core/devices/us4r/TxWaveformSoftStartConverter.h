#ifndef ARRUS_CORE_DEVICES_US4R_TXWAVEFORMSOFTSTARTCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_TXWAVEFORMSOFTSTARTCONVERTER_H

#include <vector>
#include <optional>

#include "arrus/core/api/ops/us4r/Waveform.h"
#include "arrus/core/api/ops/us4r/Pulse.h"
#include "arrus/common/asserts.h"
#include "arrus/common/utils.h"


namespace arrus::devices {


/**
 * For more information about the soft-start conversion, please refer to the current software architecture/requirements
 * documentation.
 */
class TxWaveformSoftStartConverter {
public:
    using Waveform = ::arrus::ops::us4r::Waveform;
    using WaveformBuilder = ::arrus::ops::us4r::WaveformBuilder;
    using WaveformSegment = ::arrus::ops::us4r::WaveformSegment;
    using Pulse = ::arrus::ops::us4r::Pulse;

    TxWaveformSoftStartConverter(
        uint32_t nCycles, float ts, const std::vector<float> &dutyCycles,
        const std::vector<float> &dutyCycleDurationFractions)
        : nCycles(nCycles), ts(ts) {

        this->dutyCycles = dutyCycles;
        // 100 % duty cycle at the end of the waveform.
        this->dutyCycles.push_back(1.0f);

        for(const auto fraction: dutyCycleDurationFractions) {
            this->dutyCycleDurationFractions.push_back(fraction);
        }
        // The 100% duty cycle should take all the rest of the waveform.
        this->dutyCycleDurationFractions.push_back(std::nullopt);
    }

    /**
     * Converts the given waveform with to a new waveform with the modified duty cycles.
     *
     * Converts only the segments that nRepetitions >= max nCycles.
     */
    Waveform convert(const Waveform& waveform) {
        WaveformBuilder builder;
        for(size_t i = 0; i < waveform.getSegments().size(); ++i) {
            const auto &segment = waveform.getSegments().at(i);
            const auto nRepetitions = ARRUS_SAFE_CAST(waveform.getNRepetitions().at(i), uint32_t);
            if(nRepetitions >= nCycles) {
                Waveform wf = convertToSoftStart(segment, nRepetitions);
                builder.add(wf);
            }
            else {
                // copy the segment unmodified
                builder.add(segment, nRepetitions);
            }
        }
        return builder.build();
    }

    bool apply(const Waveform& waveform) {
        const auto pulse = Pulse::fromWaveform(waveform);
        return pulse.has_value() && std::floor(pulse.value().getNPeriods()) >= nCycles && pulse.value().getCenterFrequency() >= 1e6;
    }

private:
    Waveform convertToSoftStart(const WaveformSegment &segment, const unsigned long nReps) {
        // divide the segment into multiple segments
        // Validate
        ARRUS_REQUIRES_EQUAL(segment.getState().size(), 2,
                             std::runtime_error("Soft-start requires the first segment to have exactly two states."));
        ARRUS_REQUIRES_EQUAL(segment.getDuration().at(0), segment.getDuration().at(1),
                             std::runtime_error("Soft-start requires the first two states to have exactly the same "
                                                "duration."));
        ARRUS_REQUIRES_EQUAL(segment.getState().at(0), -1*segment.getState().at(1),
                             std::runtime_error("Soft-start requires the same amplitude for both states."));

        // Calculate
        WaveformBuilder builder;
        const auto period = segment.getDuration().at(0)*2;
        auto nRepsLeft = ARRUS_SAFE_CAST((intmax_t)nReps, long);
        // We apply no more than ts of soft-start segment.
        auto nRepsSoftStartLeft = ARRUS_SAFE_CAST(std::floor(ts/period), long);

        for(size_t i = 0; i < dutyCycles.size() && nRepsLeft > 0; ++i) {
            const auto dutyCycle = dutyCycles.at(i);
            const auto durationFraction = dutyCycleDurationFractions.at(i);
            uint32_t newNRepeats = 0;
            if(!durationFraction.has_value()) {
                // Use all the rest of the number of repeats for the 100% duty cycle.
                newNRepeats = nRepsLeft;
            }
            else {
                auto expectedNRepeatsForSegment = ARRUS_SAFE_CAST(std::roundf(ts*durationFraction.value()/period), uint32_t);
                newNRepeats = std::min(ARRUS_SAFE_CAST((intmax_t)expectedNRepeatsForSegment, long), nRepsSoftStartLeft);
                nRepsSoftStartLeft -= newNRepeats;
            }
            newNRepeats = std::min<long>(ARRUS_SAFE_CAST((intmax_t)newNRepeats, long), nRepsLeft);
            const auto newSegment = getSegmentForDutyCycle(segment, dutyCycle);
            builder.add(newSegment, std::min<long>(ARRUS_SAFE_CAST((intmax_t)newNRepeats, long), nRepsLeft));

            nRepsLeft -= newNRepeats;
        }
        return builder.build();
    }

    WaveformSegment getSegmentForDutyCycle(const WaveformSegment &sourceSegment, const float dutyCycle) {
        if(dutyCycle == 1.0f) {
            return sourceSegment;
        }

        const auto halfPulseDuration = sourceSegment.getDuration().at(0);
        const float ta = halfPulseDuration*dutyCycle;
        const float tc = halfPulseDuration - ta;

        return WaveformSegment {
            {
                ta, // +/-
                tc, // CLAMP
                ta, // +/-
                tc  // CLAMP
            },
            {
                sourceSegment.getState().at(0), // +/-
                0, // CLAMP
                sourceSegment.getState().at(1), // -/+
                0  // CLAMP
            },
        };
    }

    /** Minimum number of cycles the given TX pulse should have in order to make it applicable for conversion. */
    uint32_t nCycles;
    /** Soft-start duration */
    float ts;
    /** Duty cycles to apply while soft-start is performed; dutyCycles[i] is a duty cycle for i-th soft-start segment.
     *  Values [0, 1]. */
    std::vector<float> dutyCycles;
    /** Duty cycle duration, i.e. t_i, but provided as a fraction of t_i. */
    std::vector<std::optional<float>> dutyCycleDurationFractions;
};

}

#endif//ARRUS_CORE_DEVICES_US4R_TXWAVEFORMSOFTSTARTCONVERTER_H
