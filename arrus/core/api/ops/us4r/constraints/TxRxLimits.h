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
    TxRxLimits(TxLimits tx1, TxLimits tx2, RxLimits rx, const Interval<float> &pri) : tx1(std::move(tx1)), tx2(std::move(tx2)), rx(std::move(rx)), pri(pri) {}

    /**
     * TX limits for HV rail amplitude 1, i.e. HVM/P 1.
     */
    const TxLimits &getTx1() const { return tx1; }

    /**
     * TX limits for HV rail amplitude 2, i.e. HVM/P 0.
     */
    const TxLimits &getTx2() const { return tx2; }


    const RxLimits &getRx() const { return rx; }
    const Interval<float> &getPri() const { return pri; }

private:
    TxLimits tx1;
    TxLimits tx2;
    RxLimits rx;
    Interval<float> pri;
};

}

#endif //ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXLIMITS_H
