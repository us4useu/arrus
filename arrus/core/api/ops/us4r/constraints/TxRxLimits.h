#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXLIMITS_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXLIMITS_H

#include "RxLimits.h"

#include <utility>
#include "TxLimits.h"
namespace arrus::ops::us4r {

/**
 * TxRx op limits (constraints on the parameter values).
 */
class TxRxLimits {
public:
    TxRxLimits(TxLimits tx, RxLimits rx, const Interval<float> &pri) : tx(std::move(tx)), rx(std::move(rx)), pri(pri) {}

    const TxLimits &getTx() const { return tx; }
    const RxLimits &getRx() const { return rx; }
    const Interval<float> &getPri() const { return pri; }

private:
    TxLimits tx;
    RxLimits rx;
    Interval<float> pri;
};

}

#endif //ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXLIMITS_H
