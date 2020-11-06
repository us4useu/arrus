#ifndef ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
#define ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H

#include <utility>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/framework.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/Rx.h"

namespace arrus::ops::us4r {

// TODO use pair instead
class TxRx {
public:
    // TODO(pjarosik) remove default constructor!!! Currently required by py swig wrapper
    TxRx()
    :tx(std::vector<bool>{}, std::vector<float>{}, Pulse(0, 0, false)),
     rx(std::vector<bool>{}, std::make_pair<unsigned int, unsigned int>((unsigned int)0, (unsigned int)0)),
     pri(0.0f)
    {}

    TxRx(Tx tx, Rx rx, float pri) : tx(std::move(tx)), rx(std::move(rx)), pri(pri) {}

    const Tx &getTx() const {
        return tx;
    }

    const Rx &getRx() const {
        return rx;
    }

    float getPri() const {
        return pri;
    }

private:
    Tx tx;
    Rx rx;
    float pri;
};

class TxRxSequence {
public:
    TxRxSequence(
        std::vector<TxRx> sequence, TGCCurve tgcCurve)
        : txrxs(std::move(sequence)), tgcCurve(std::move(tgcCurve)) {}

    const std::vector<TxRx> &getOps() const {
        return txrxs;
    }

    const TGCCurve &getTgcCurve() const {
        return tgcCurve;
    }

private:
    std::vector<TxRx> txrxs;
    TGCCurve tgcCurve;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
