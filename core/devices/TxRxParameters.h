#ifndef ARRUS_CORE_DEVICES_TXRXPARAMETERS_H
#define ARRUS_CORE_DEVICES_TXRXPARAMETERS_H

#include <gsl/gsl>
#include <utility>
#include <ostream>

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/types.h"
#include "arrus/common/format.h"
#include "arrus/core/api/ops/us4r/Pulse.h"

namespace arrus::devices {

class TxRxParameters {
public:
    /** (us4oem module, iteration) -> return true if can continue data acquisition */
    using SequenceCallback = std::function<void(Ordinal, uint16)>;

    static const TxRxParameters US4OEM_NOP;

    static TxRxParameters createRxNOPCopy(const TxRxParameters& op) {
        return TxRxParameters(
            op.txAperture,
            op.txDelays,
            op.txPulse,
            BitMask(op.rxAperture.size(), false),
            op.rxSampleRange,
            op.rxDecimationFactor,
            op.pri,
            op.rxPadding
        );
    }

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
     * @param rxSampleRange [start, end) range of samples to acquire, starts from 0
     * @param rxDecimationFactor
     * @param pri
     * @param rxPadding how many 0-channels padd from the left and right
     * @param checkpoint true if the given tx/rx should be a checkpoint
     *   (i.e. the device should stop and wait for TriggerSync)
     * @param callback a callback that should be called when the data acquisition
     *   at this point ends. When set to  != nullptr, the event will be set to syncReq=true,
     *   and, after calling the callback function; master module will require TriggerSync
     */
    TxRxParameters(std::vector<bool> txAperture,
                   std::vector<float> txDelays,
                   const ops::us4r::Pulse &txPulse,
                   std::vector<bool> rxAperture,
                   Interval<uint32> rxSampleRange,
                   uint32 rxDecimationFactor, float pri,
                   Tuple<ChannelIdx> rxPadding = {0, 0},
                   bool checkpoint = false,
                   std::optional<SequenceCallback> callback = std::nullopt,
                   std::optional<SequenceCallback> pciCallback = std::nullopt)
        : txAperture(std::move(txAperture)), txDelays(std::move(txDelays)),
          txPulse(txPulse),
          rxAperture(std::move(rxAperture)), rxSampleRange(std::move(rxSampleRange)),
          rxDecimationFactor(rxDecimationFactor), pri(pri),
          rxPadding(std::move(rxPadding)),
          checkpoint(checkpoint),
          callback(std::move(callback)),
          pciCallback(std::move(pciCallback)) {}

    [[nodiscard]] const std::vector<bool> &getTxAperture() const {
        return txAperture;
    }

    [[nodiscard]] const std::vector<float> &getTxDelays() const {
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

    [[nodiscard]] uint32 getNumberOfSamples() const {
        return rxSampleRange.end() - rxSampleRange.start();
    }

    [[nodiscard]] int32 getRxDecimationFactor() const {
        return rxDecimationFactor;
    }

    [[nodiscard]] float getPri() const {
        return pri;
    }

    [[nodiscard]] const Tuple<ChannelIdx> &getRxPadding() const {
        return rxPadding;
    }

    [[nodiscard]] const std::optional<SequenceCallback> &getCallback() const {
        return callback;
    }

    // TODO try keeping this class immutable
    void setCallback(const std::optional<SequenceCallback> &clbk) {
        this->callback = clbk;
    }

    void setCheckpoint(bool ckpt) {
        this->checkpoint = ckpt;
    }

    bool isCheckpoint() const {
        return checkpoint;
    }

    [[nodiscard]] bool isNOP() const  {
        auto atLeastOneTxActive = std::reduce(
            std::begin(txAperture),
            std::end(txAperture),
            false, [](auto a, auto b) {return a | b;});
        auto atLeastOneRxActive = std::reduce(
            std::begin(rxAperture),
            std::end(rxAperture),
            false, [](auto a, auto b) {return a | b;});
        return !atLeastOneTxActive && !atLeastOneRxActive;
    }

    [[nodiscard]] bool isRxNOP() const {
        auto atLeastOneRxActive = std::reduce(
            std::begin(rxAperture),
            std::end(rxAperture),
            false, [](auto a, auto b) {return a | b;});
        return !atLeastOneRxActive;
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
        os << ", fs divider: " << parameters.getRxDecimationFactor();
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
    ::std::vector<float> txDelays;
    ::arrus::ops::us4r::Pulse txPulse;
    ::std::vector<bool> rxAperture;
    // TODO change to a simple pair
    Interval<uint32> rxSampleRange;
    int32 rxDecimationFactor;
    float pri;
    Tuple<ChannelIdx> rxPadding;
    bool checkpoint;
    std::optional<SequenceCallback> callback;
    std::optional<SequenceCallback> pciCallback;
};

using TxRxParamsSequence = std::vector<TxRxParameters>;

/**
 * Returns the number of actual ops, that is, a the number of ops excluding RxNOPs.
 */
uint16 getNumberOfNoRxNOPs(const TxRxParamsSequence &seq);

}

#endif //ARRUS_CORE_DEVICES_TXRXPARAMETERS_H
