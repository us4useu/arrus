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
    static constexpr float NO_VALUE = -1;
    /**
     * Tx/Rx sequence to execute on Us4R device.
     *
     * @param sequence a list of tx/rxs that compose a given sequence
     * @param tgcCurve tgc curve to apply
     * @param sri sequence repetition interval - the total time that a given sequence
     *  of txrxs should take. Should be not smaller than the total Pulse Repetition Interval
     *  of transmits performed in the sequence. Optional; if not set, the sequence repetition
     *  interval will be determined by total pulse repetition interval.
     * @param nRepeats number of repeats: how many times the sequence should be repeated on the device
     * @param bri batch repetition interval - the total time that a given number of repetition
     *  of the sequence should take. Should be not smaller than time required to execute the
     *  sequence multiplied by the number of repeats. Optional; if not set, BRI will be determined based
     *  on total sequence execution time.
     */
    TxRxSequence(std::vector<TxRx> sequence, TGCCurve tgcCurve, float sri = NO_VALUE,
                 unsigned short nRepeats = 1, float bri = NO_VALUE)
        : txrxs(std::move(sequence)), tgcCurve(std::move(tgcCurve)), sri(sri),
        nRepeats(nRepeats), bri(bri) {}

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
     * Returns sequence repetition interval (the total time the given sequence should actually take).
     * nullopt means that the SRI will be equal total PRI of the performed TxRxs.
     */
    const std::optional<float> getSri() const {
        if(sri.value() != NO_VALUE) {
            return sri;
        }
        else {
            return std::optional<float>();
        }
    }

    const std::optional<float> getBri() const {
        if(sri.value() != NO_VALUE) {
            return sri;
        }
        else {
            return std::optional<float>();
        }
    }

    unsigned short getNRepeats() const {
        return nRepeats;
    }


private:
    std::vector<TxRx> txrxs;
    TGCCurve tgcCurve;
    std::optional<float> sri;
    unsigned short nRepeats;
    std::optional<float> bri;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
