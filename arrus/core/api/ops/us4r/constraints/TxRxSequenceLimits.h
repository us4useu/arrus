#ifndef ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXSEQUENCELIMITS_H
#define ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXSEQUENCELIMITS_H

#include "TxRxLimits.h"
#include "arrus/core/api/common/types.h"
namespace arrus::ops::us4r {

class TxRxSequenceLimitsBuilder;

/**
 * TX/RX sequence limits.
 */
class TxRxSequenceLimits {
public:
    TxRxSequenceLimits(const TxRxLimits &txrx, const Interval<uint32> &size, uint32 maxNFirings)
        : txrx(txrx), size(size), maxNFirings(maxNFirings) {}

    /** Assumes that the sequence size == max number of firings. */
    TxRxSequenceLimits(const TxRxLimits &txrx, const Interval<uint32> &size): TxRxSequenceLimits(txrx, size, size.end()) {}

    const TxRxLimits &getTxRx() const { return txrx; }
    const Interval<uint32> &getSize() const { return size; }
    uint32 getMaxNumberOfFirings() const { return maxNFirings; }

private:
    friend class TxRxSequenceLimitsBuilder;
    TxRxLimits txrx;
    Interval<uint32> size;
    /** Maximum number of different TX/RX operators.
     * It should be equal to the min(max rx apertures, max tx apertures, max tx delays). */
    uint32 maxNFirings;
};

class TxRxSequenceLimitsBuilder {
public:
    explicit TxRxSequenceLimitsBuilder(TxRxSequenceLimits limits): limits(std::move(limits)) {}

    TxRxSequenceLimitsBuilder& setTxRxLimits(const TxLimits &txLimits, const RxLimits &rxLimits,
                                          const Interval<float> &pri) {
        limits->txrx = TxRxLimits{txLimits, rxLimits, pri};
        return *this;
    }

    TxRxSequenceLimits build() {
        return limits.value();
    }

private:
    std::optional<TxRxSequenceLimits> limits;
};



}

#endif//ARRUS_ARRUS_CORE_API_OPS_US4R_CONSTRAINTS_TXRXSEQUENCELIMITS_H
