#ifndef ARRUS_CORE_DEVICES_TXRXPARAMETERS_H
#define ARRUS_CORE_DEVICES_TXRXPARAMETERS_H

#include <gsl/gsl>
#include <utility>
#include <ostream>

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/common/format.h"
#include "arrus/core/api/ops/us4r/Pulse.h"

namespace arrus::devices {

class TxRxParameters {
public:
    static const TxRxParameters NOP;

    /**
     *
     * ** tx aperture, tx delays and rx aperture should have the same size
     * (tx delays is NOT limited to the tx aperture active elements -
     * the whole array must be provided).**
     *
     * @param txAperture
     * @param txDelays
     * @param txPulse
     * @param rxAperture
     * @param rxSampleRange
     * @param rxDecimationFactor
     * @param pri
     */
    TxRxParameters(std::vector<bool> txAperture,
                   const gsl::span<const float> &txDelays,
                   const ops::us4r::Pulse &txPulse,
                   std::vector<bool> rxAperture,
                   const Interval<uint32> &rxSampleRange,
                   uint32 rxDecimationFactor, float pri)
        : txAperture(std::move(txAperture)), txDelays(txDelays),
          txPulse(txPulse),
          rxAperture(std::move(rxAperture)), rxSampleRange(rxSampleRange),
          rxDecimationFactor(rxDecimationFactor), pri(pri) {}

    [[nodiscard]] const std::vector<bool> &getTxAperture() const {
        return txAperture;
    }

    [[nodiscard]] const gsl::span<const float> &getTxDelays() const {
        return txDelays;
    }

    [[nodiscard]] const ops::us4r::Pulse &getTxPulse() const {
        return txPulse;
    }

    [[nodiscard]] const std::vector<bool> &getRxAperture() const {
        return rxAperture;
    }

    [[nodiscard]] const Interval<uint32> &getRxSampleRange() const {
        return rxSampleRange;
    }

    [[nodiscard]] int32 getRxDecimationFactor() const {
        return rxDecimationFactor;
    }

    [[nodiscard]] float getPri() const {
        return pri;
    }

    bool isNOP() {
        auto atLeastOneTxActive = std::reduce(
            std::begin(txAperture),
            std::end(txAperture),
            false, [](auto a, auto b) {return a | b;});
        auto atLeastOneRxActive = std::reduce(
            std::begin(rxAperture),
            std::end(rxAperture),
            false, [](auto a, auto b) {return a | b;});
        return atLeastOneTxActive || atLeastOneRxActive;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const TxRxParameters &parameters) {
        os << "Tx/Rx: ";
        os << "TX: ";
        os << "aperture: " << ::arrus::toString(parameters.getTxAperture())
           << ", delays: " << ::arrus::toString(parameters.getTxDelays())
           << ", center frequency: " << parameters.getTxPulse().getCenterFrequency()
           << ", n. periods: " << parameters.getTxPulse().getNPeriods()
           << ", inverse: " << parameters.getTxPulse().isInverse();
        os << "; RX: ";
        os << "aperture: " << ::arrus::toString(parameters.getRxAperture());
        os << "sample range: " << parameters.getRxSampleRange().start() << ", "
           << parameters.getRxSampleRange().end();
        os << "fs divider: " << parameters.getRxDecimationFactor();
        os << std::endl;
        return os;
    }

    bool operator==(const TxRxParameters &rhs) const {
        return txAperture == rhs.txAperture &&
               txDelays == rhs.txDelays &&
               txPulse == rhs.txPulse &&
               rxAperture == rhs.rxAperture &&
               rxSampleRange == rhs.rxSampleRange &&
               rxDecimationFactor == rhs.rxDecimationFactor &&
               pri == rhs.pri;
    }

    bool operator!=(const TxRxParameters &rhs) const {
        return !(rhs == *this);
    }
private:
    ::std::vector<bool> txAperture;
    ::gsl::span<const float> txDelays;
    ::arrus::ops::us4r::Pulse txPulse;
    ::std::vector<bool> rxAperture;
    Interval<uint32> rxSampleRange;
    int32 rxDecimationFactor;
    float pri;
};

using TxRxParamsSequence = std::vector<TxRxParameters>;
}

#endif //ARRUS_CORE_DEVICES_TXRXPARAMETERS_H
