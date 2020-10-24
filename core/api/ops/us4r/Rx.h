#ifndef ARRUS_CORE_API_OPS_US4R_RX_H
#define ARRUS_CORE_API_OPS_US4R_RX_H

#include <utility>

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

class Rx {
public:
    Rx(BitMask aperture, const Interval<uint32> &sampleRange, double fsDivider)
        : aperture(std::move(aperture)), sampleRange(sampleRange),
          fsDivider(fsDivider) {}

    [[nodiscard]] const BitMask &getAperture() const {
        return aperture;
    }

    [[nodiscard]] const Interval<uint32> &getSampleRange() const {
        return sampleRange;
    }

    [[nodiscard]] double getFsDivider() const {
        return fsDivider;
    }

private:
    BitMask aperture;
    Interval<uint32> sampleRange;
    double fsDivider;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_RX_H
