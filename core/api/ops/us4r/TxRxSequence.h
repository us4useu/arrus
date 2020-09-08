#ifndef ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
#define ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H

#include <utility>
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/Rx.h"

namespace arrus::ops::us4r {

class TxRxSequence {
public:
    using TxRx = std::pair<Tx, Rx>;

    TxRxSequence(std::vector<TxRx> sequence, double pri)
    : sequence(std::move(sequence)), pri(pri) {}

    [[nodiscard]] const std::vector<TxRx> &getSequence() const {
        return sequence;
    }

    [[nodiscard]] double getPri() const {
        return pri;
    }

private:

    std::vector<TxRx> sequence;
    double pri;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
