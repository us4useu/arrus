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
    TxRxLimits(TxLimits tx0, TxLimits tx1, RxLimits rx, const Interval<float> &pri) : tx0(std::move(tx0)), tx1(std::move(tx1)), rx(std::move(rx)), pri(pri) {}

    const TxLimits &getTxHV0() const { return tx0; }
    const TxLimits &getTxHV1() const { return tx1; }
    const RxLimits &getRx() const { return rx; }
    const Interval<float> &getPri() const { return pri; }

private:
    TxLimits tx0;
    TxLimits tx1;
    RxLimits rx;
    Interval<float> pri;
};

}

#endif //ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXLIMITS_H
