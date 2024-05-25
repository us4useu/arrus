#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_RXLIMITS_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_RXLIMITS_H

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

/**
 * RX op limits.
 *
 * The instance of this class defines what constraints are applied on the RX op parameters.
 */
class RxLimits {
public:
    explicit RxLimits(const Interval<uint32> &nSamples) : nSamples(nSamples) {}

    const Interval<uint32> &getNSamples() const { return nSamples; }

private:
    Interval<uint32> nSamples;
};

}// namespace arrus::ops::us4r

#endif//ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_RXLIMITS_H
