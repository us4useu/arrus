#ifndef ARRUS_CORE_API_OPS_US4R_RX_H
#define ARRUS_CORE_API_OPS_US4R_RX_H

#include <utility>

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

/**
 * An operation that performs a single data reception (Rx).
 */
class Rx {
public:
    /**
     * Rx constructor.
     *
     * @param aperture receive aperture to use;
     *  aperture[i] = true means that the i-th channel should be turned on
     * @param rxSampleRange [start, end) range of samples to acquire, starts from 0
     * @param downsamplingFactor the factor by which the sampling frequency should be divided, an integer
     */
    Rx(BitMask aperture, Interval<uint32> sampleRange,
       uint32 downsamplingFactor = 1, Tuple<ChannelIdx> padding = {0, 0})
        : aperture(std::move(aperture)), sampleRange(std::move(sampleRange)),
          downsamplingFactor(downsamplingFactor),
          padding(std::move(padding)) {}

    const BitMask &getAperture() const {
        return aperture;
    }

    const Interval<uint32> &getSampleRange() const {
        return sampleRange;
    }

    uint32 getDownsamplingFactor() const {
        return downsamplingFactor;
    }

    const Tuple<ChannelIdx> &getPadding() const {
        return padding;
    }

private:
    BitMask aperture;
    Interval<uint32> sampleRange;
    uint32 downsamplingFactor;
    Tuple<ChannelIdx> padding;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_RX_H
