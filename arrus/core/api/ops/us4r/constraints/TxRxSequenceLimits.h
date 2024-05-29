#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXSEQUENCELIMITS_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXSEQUENCELIMITS_H

#include "TxRxLimits.h"
#include "arrus/core/api/common/types.h"
namespace arrus::ops::us4r {

/**
 * TX/RX sequence limits.
 */
class TxRxSequenceLimits {
public:
    TxRxSequenceLimits(const TxRxLimits &txrx, const Interval<uint32> &size) : txrx(txrx), size(size) {}

    const TxRxLimits &getTxRx() const { return txrx; }
    const Interval<uint32> &getSize() const { return size; }

private:
    TxRxLimits txrx;
    Interval<uint32> size;
};

}

#endif//ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXSEQUENCELIMITS_H
