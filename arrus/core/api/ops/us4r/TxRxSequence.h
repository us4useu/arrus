#ifndef ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
#define ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H

#include <utility>
#include <optional>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/framework.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/Rx.h"

namespace arrus::ops::us4r {

class TxRx {
public:
    // TODO(pjarosik) remove default constructor!!! Currently required by py swig wrapper
    TxRx()
    :tx(std::vector<bool>{}, std::vector<float>{},
        Pulse(0, 0, false)),
     rx(std::vector<bool>{},
        std::make_pair<unsigned int, unsigned int>((unsigned int)0, (unsigned int)0)),
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
    /**
     * Tx/Rx sequence to execute on Us4R device.
     *
     * @param sequence a list of tx/rxs that compose a given sequence
     * @param tgcCurve tgc curve to apply
     * @param sri frame repetition interval - the total time that a given sequence should take. Should be not smaller
     */
    TxRxSequence(std::vector<TxRx> sequence, TGCCurve tgcCurve, std::optional<float> sri)
        : txrxs(std::move(sequence)), tgcCurve(std::move(tgcCurve)), sri(sri) {}

    /**
     * Sequence of operations to perform.
     */
    const std::vector<TxRx> &getOps() const {
        return txrxs;
    }

    /**
     * Initial TGC curve points.
     */
    const TGCCurve &getTgcCurve() const {
        return tgcCurve;
    }

    /**
     * Returns frame repetition interval (the total time the given sequence should actually take).
     * nullopt means that the frame acquistion time should be determined by total PRI only.
     */
    const std::optional<float> &getSri() const {
        return sri;
    }

private:
    std::vector<TxRx> txrxs;
    TGCCurve tgcCurve;
    std::optional<float> sri;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
