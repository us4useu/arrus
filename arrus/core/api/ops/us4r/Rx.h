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
     * @param placement probe on which the RX should be performed.
     */
    Rx(std::vector<bool> aperture, std::pair<unsigned int, unsigned int> sampleRange,
       unsigned int downsamplingFactor = 1,
       std::pair<unsigned short, unsigned short> padding = {(ChannelIdx) 0, (ChannelIdx) 0},
       devices::DeviceId placement = devices::DeviceId(devices::DeviceType::Probe, 0))
        : aperture(std::move(aperture)), sampleRange(std::move(sampleRange)), downsamplingFactor(downsamplingFactor),
          padding(std::move(padding)), placement(placement) {
        ARRUS_REQUIRES_TRUE_IAE(placement.getDeviceType() == devices::DeviceType::Probe,
                                "Only probe can be set as a placement for TX.");
    }

    const std::vector<bool> &getAperture() const { return aperture; }

    const std::pair<unsigned, unsigned> &getSampleRange() const { return sampleRange; }

    unsigned getDownsamplingFactor() const { return downsamplingFactor; }

    const std::pair<unsigned short, unsigned short> &getPadding() const { return padding; }

    devices::DeviceId getPlacement() const { return placement; }

private:
    std::vector<bool> aperture;
    std::pair<unsigned, unsigned> sampleRange;
    unsigned downsamplingFactor;
    std::pair<unsigned short, unsigned short> padding;
    devices::DeviceId placement;
};

}// namespace arrus::ops::us4r

#endif//ARRUS_CORE_API_OPS_US4R_RX_H
