#ifndef ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
#define ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H

#include <utility>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/framework.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/Rx.h"

namespace arrus::ops::us4r {

class TxRxSequence {
public:
    using TxRx = std::pair<Tx, Rx>;

    TxRxSequence(
        std::vector<TxRx> sequence,
        float pri, TGCCurve tgcCurve)
        : txrxs(std::move(sequence)), pri(pri),
          tgcCurve(std::move(tgcCurve)) {}

    [[nodiscard]] const std::vector<TxRx> &getOps() const {
        return txrxs;
    }

    [[nodiscard]] float getPri() const {
        return pri;
    }

    [[nodiscard]] const TGCCurve &getTgcCurve() const {
        return tgcCurve;
    }

private:
    std::vector<TxRx> txrxs;
    float pri;
    TGCCurve tgcCurve;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
