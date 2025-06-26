#ifndef ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
#define ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H

#include <optional>
#include <utility>
#include <algorithm>
#include <iterator>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/framework.h"
#include "arrus/core/api/ops/us4r/Rx.h"
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/tgc.h"

#include <unordered_set>

namespace arrus::ops::us4r {

/**
 * A single tx/rx operation to perform.
 */
class TxRx {
public:
    TxRx()
        : tx(std::vector<bool>{}, std::vector<float>{}, Pulse(0, 0, false)),
          rx(std::vector<bool>{}, std::make_pair<unsigned int, unsigned int>((unsigned int) 0, (unsigned int) 0)),
          pri(0.0f) {}

    /**
     * TxRx constructor.
     *
     * @param tx - tx description
     * @param rx - rx description
     * @param pri - pulse repetition interval
     */
    TxRx(Tx tx, Rx rx, float pri) : tx(std::move(tx)), rx(std::move(rx)), pri(pri) {}

    const Tx &getTx() const { return tx; }

    const Rx &getRx() const { return rx; }

    float getPri() const { return pri; }

private:
    Tx tx;
    Rx rx;
    float pri;
};

class TxRxSequence {
public:
    static constexpr float NO_SRI = -1;
    /**
     * Tx/Rx sequence to execute on Us4R device.
     *
     * @param sequence a list of tx/rxs that compose a given sequence
     * @param tgcCurve tgc curve to apply
     * @param sri sequence repetition interval - the total time that a given sequence should take.
     * @param nRepeats - the number of repetitions of a given sequence. Determines the size of the batch
     */
    TxRxSequence(std::vector<TxRx> sequence, TGCCurve tgcCurve, float sri = NO_SRI, int16 nRepeats = 1)
        : txrxs(std::move(sequence)), tgcCurve(std::move(tgcCurve)), sri(sri), nRepeats(nRepeats) {}

    TxRxSequence copy(std::vector<TxRx> ops) {
        return TxRxSequence(std::move(ops), this->tgcCurve, this->sri.value(), this->nRepeats);
    }

    /**
     * Returns vector of operations to perform.
     */
    const std::vector<TxRx> &getOps() const { return txrxs; }

    /**
     * Initial TGC curve points.
     */
    const TGCCurve &getTgcCurve() const { return tgcCurve; }

    /**
     * Returns sequence repetition interval (the total time the given sequence should actually take).
     * nullopt means that the frame acquisition time should be determined by total PRI only.
     */
    const std::optional<float> getSri() const {
        if (sri.value() != NO_SRI) {
            return sri;
        } else {
            return std::optional<float>();
        }
    }

    int16 getNRepeats() const { return nRepeats; }

    /**
     * Returns the ordinal number of the probe used for RX. If the RX probe is not-unique,
     * this method will throw IllegalStateException.
     */
    devices::DeviceId getRxProbeId() const {
        std::unordered_set<devices::Ordinal> p;
        auto toRxProbe = [](const TxRx &op) { return op.getRx().getPlacement().getOrdinal(); };
        std::transform(std::begin(txrxs), std::end(txrxs), std::inserter(p, std::begin(p)), toRxProbe);
        if(p.size() != 1) {
            throw IllegalStateException("There should be exactly one RX probe in the sequence.");
        }
        devices::Ordinal ordinal = *std::begin(p);
        return devices::DeviceId{devices::DeviceType::Probe, ordinal};
    }
    /**
     * Returns the ordinal number of the probe used for RX. If the RX probe is not-unique,
     * this method will throw IllegalStateException.
     */
    devices::DeviceId getTxProbeId() const {
        std::unordered_set<devices::Ordinal> p;
        auto toTxProbe = [](const TxRx &op) { return op.getTx().getPlacement().getOrdinal(); };
        std::transform(std::begin(txrxs), std::end(txrxs), std::inserter(p, std::begin(p)), toTxProbe);
        if(p.size() != 1) {
            throw IllegalStateException("There should be exactly one TX probe in the sequence.");
        }
        devices::Ordinal ordinal = *std::begin(p);
        return devices::DeviceId{devices::DeviceType::Probe, ordinal};
    }

private:
    std::vector<TxRx> txrxs;
    TGCCurve tgcCurve;
    std::optional<float> sri;
    int16 nRepeats;
};

}// namespace arrus::ops::us4r

#endif//ARRUS_CORE_API_OPS_US4R_TXRXSEQUENCE_H
