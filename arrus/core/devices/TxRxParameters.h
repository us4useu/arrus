#ifndef ARRUS_CORE_DEVICES_TXRXPARAMETERS_H
#define ARRUS_CORE_DEVICES_TXRXPARAMETERS_H

#include <gsl/gsl>
#include <ostream>
#include <utility>

#include "arrus/common/format.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/ops/us4r/Pulse.h"
#include "arrus/core/common/aperture.h"
#include "arrus/core/common/collections.h"

namespace arrus::devices::us4r {

class TxRxParameters {
public:
    static const TxRxParameters US4OEM_NOP;

    static TxRxParameters createRxNOPCopy(const TxRxParameters &op) {
        return TxRxParameters(op.txAperture, op.txDelays, op.txPulse, BitMask(op.rxAperture.size(), false),
                              op.rxSampleRange, op.rxDecimationFactor, op.pri, op.rxPadding, op.rxDelay,
                              op.bitstreamId, op.maskedChannelsTx, op.maskedChannelsRx);
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
     */
    TxRxParameters(std::vector<bool> txAperture, std::vector<float> txDelays, const ops::us4r::Pulse &txPulse,
                   std::vector<bool> rxAperture, Interval<uint32> rxSampleRange, int32 rxDecimationFactor, float pri,
                   Tuple<ChannelIdx> rxPadding = {0, 0}, float rxDelay = 0.0f,
                   std::optional<BitstreamId> bitstreamId = std::nullopt,
                   std::unordered_set<ChannelIdx> maskedChannelsTx = {},
                   std::unordered_set<ChannelIdx> maskedChannelsRx = {}
    )
        : txAperture(std::move(txAperture)), txDelays(std::move(txDelays)), txPulse(txPulse),
          rxAperture(std::move(rxAperture)), rxSampleRange(std::move(rxSampleRange)),
          rxDecimationFactor(rxDecimationFactor), pri(pri), rxPadding(std::move(rxPadding)), rxDelay(rxDelay),
          bitstreamId(bitstreamId), maskedChannelsTx(maskedChannelsTx), maskedChannelsRx(maskedChannelsRx) {}

    [[nodiscard]] const std::vector<bool> &getTxAperture() const { return txAperture; }

    [[nodiscard]] const std::vector<float> &getTxDelays() const { return txDelays; }

    [[nodiscard]] const ops::us4r::Pulse &getTxPulse() const { return txPulse; }

    [[nodiscard]] const std::vector<bool> &getRxAperture() const { return rxAperture; }

    [[nodiscard]] const Interval<uint32> &getRxSampleRange() const { return rxSampleRange; }

    [[nodiscard]] uint32 getNumberOfSamples() const { return rxSampleRange.end() - rxSampleRange.start(); }

    [[nodiscard]] int32 getRxDecimationFactor() const { return rxDecimationFactor; }

    [[nodiscard]] float getPri() const { return pri; }

    [[nodiscard]] const Tuple<ChannelIdx> &getRxPadding() const { return rxPadding; }

    [[nodiscard]] bool isNOP() const {
        auto atLeastOneTxActive =
            ::arrus::reduce(std::begin(txAperture), std::end(txAperture), false, [](auto a, auto b) { return a | b; });
        auto atLeastOneRxActive =
            ::arrus::reduce(std::begin(rxAperture), std::end(rxAperture), false, [](auto a, auto b) { return a | b; });
        return !atLeastOneTxActive && !atLeastOneRxActive;
    }

    [[nodiscard]] bool isRxNOP() const {
        auto atLeastOneRxActive =
            ::arrus::reduce(std::begin(rxAperture), std::end(rxAperture), false, [](auto a, auto b) { return a | b; });
        return !atLeastOneRxActive;
    }

    float getRxDelay() const { return rxDelay; }

    const std::optional<BitstreamId> &getBitstreamId() const { return bitstreamId; }

    // TODO(pjarosik) consider removing the below setter (keep this class immutable).
    void setRxDelay(float delay) { this->rxDelay = delay; }

    [[nodiscard]] const std::unordered_set<ChannelIdx> &getMaskedChannelsTx() const { return maskedChannelsTx; }

    [[nodiscard]] const std::unordered_set<ChannelIdx> &getMaskedChannelsRx() const { return maskedChannelsRx; }

    friend std::ostream &operator<<(std::ostream &os, const TxRxParameters &parameters) {
        os << std::scientific;
        os << "Tx/Rx: ";
        os << "TX: ";
        os << "aperture: " << ::arrus::toString(parameters.getTxAperture());
        os << ", delays: ";
        for(auto d: parameters.getTxDelays()) {
            os << d << ", ";
        }
        os << ", center frequency: " << parameters.getTxPulse().getCenterFrequency()
           << ", n. periods: " << parameters.getTxPulse().getNPeriods()
           << ", inverse: " << parameters.getTxPulse().isInverse();
        os << "; RX: ";
        os << "aperture: " << ::arrus::toString(parameters.getRxAperture());
        os << ", sample range: " << parameters.getRxSampleRange().start() << ", " << parameters.getRxSampleRange().end();
        os << ", fs divider: " << parameters.getRxDecimationFactor() << ", padding: " << parameters.getRxPadding()[0]
           << ", " << parameters.getRxPadding()[1];
        os << ", rx delay: " << parameters.getRxDelay();
        if (parameters.getBitstreamId().has_value()) {
            os << ", bitstream id: " << parameters.getBitstreamId().value();
        }
        os << ", masked channels TX: ";
        for(auto ch: parameters.getMaskedChannelsTx()) {
            os << ch << ", ";
        }
        os << ", masked channels RX: ";
        for(auto ch: parameters.getMaskedChannelsRx()) {
            os << ch << ", ";
        }
        os << std::endl;
        os << std::fixed;
        return os;
    }

    bool operator==(const TxRxParameters &rhs) const {
        return txAperture == rhs.txAperture && txDelays == rhs.txDelays && txPulse == rhs.txPulse
            && rxAperture == rhs.rxAperture && rxSampleRange == rhs.rxSampleRange
            && rxDecimationFactor == rhs.rxDecimationFactor && pri == rhs.pri && rxDelay == rhs.rxDelay
            && bitstreamId == rhs.bitstreamId
            && maskedChannelsTx == rhs.maskedChannelsTx
            && maskedChannelsRx == rhs.maskedChannelsRx;
    }

    bool operator!=(const TxRxParameters &rhs) const { return !(rhs == *this); }

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
    float rxDelay;
    std::optional<BitstreamId> bitstreamId;
    /** A set of masked channels. A masked channel means that it will be used in the TX or RX,
     * however it will be present in the output RF frame, as it were a channel enabled in RX. */
    std::unordered_set<ChannelIdx> maskedChannelsTx;
    std::unordered_set<ChannelIdx> maskedChannelsRx;
};

class TxRxParametersBuilder {
public:
    explicit TxRxParametersBuilder(const TxRxParameters &params) {
        txAperture = params.getTxAperture();
        txDelays = params.getTxDelays();
        txPulse = params.getTxPulse();
        rxAperture = params.getRxAperture();
        rxSampleRange = params.getRxSampleRange();
        rxDecimationFactor = params.getRxDecimationFactor();
        pri = params.getPri();
        rxPadding = params.getRxPadding();
        rxDelay = params.getRxDelay();
        bitstreamId = params.getBitstreamId();
        maskedChannelsTx = params.getMaskedChannelsTx();
        maskedChannelsRx = params.getMaskedChannelsRx();
    }

    explicit TxRxParametersBuilder(const arrus::ops::us4r::TxRx &op) {
        auto &tx = op.getTx();
        auto &rx = op.getRx();

        Interval<uint32> sampleRange(rx.getSampleRange().first, rx.getSampleRange().second);
        Tuple<ChannelIdx> padding({rx.getPadding().first, rx.getPadding().second});

        this->txAperture = tx.getAperture();
        this->txDelays = tx.getDelays();
        this->txPulse = tx.getExcitation();
        this->rxAperture = rx.getAperture();
        this->rxSampleRange = sampleRange;
        this->rxDecimationFactor = rx.getDownsamplingFactor();
        this->pri = op.getPri();
        this->rxPadding = padding;
        this->rxDelay = 0.0f;
        this->bitstreamId = std::nullopt;
        this->maskedChannelsTx = {};
        this->maskedChannelsRx = {};
    }

    TxRxParameters build() {
        if (!txPulse.has_value()) {
            throw IllegalArgumentException("TX pulse definition is required");
        }
        return TxRxParameters(txAperture, txDelays, txPulse.value(), rxAperture, rxSampleRange, rxDecimationFactor, pri,
                              rxPadding, rxDelay, bitstreamId, maskedChannelsTx, maskedChannelsRx);
    }

    void convertToNOP() {
        txAperture = BitMask(txAperture.size(), false);
        rxAperture = BitMask(rxAperture.size(), false);
        txDelays = getNTimes<float>(0.0f, txAperture.size());
    }

    void setTxAperture(const std::vector<bool> &value) { TxRxParametersBuilder::txAperture = value; }
    void setTxDelays(const std::vector<float> &value) { TxRxParametersBuilder::txDelays = value; }
    void setTxPulse(const std::optional<::arrus::ops::us4r::Pulse> &value) { TxRxParametersBuilder::txPulse = value; }
    void setRxAperture(const std::vector<bool> &value) { TxRxParametersBuilder::rxAperture = value; }
    void setRxSampleRange(const Interval<uint32> &value) { TxRxParametersBuilder::rxSampleRange = value; }
    void setRxDecimationFactor(int32 value) { TxRxParametersBuilder::rxDecimationFactor = value; }
    void setPri(float value) { TxRxParametersBuilder::pri = value; }
    void setRxPadding(const Tuple<ChannelIdx> &value) { TxRxParametersBuilder::rxPadding = value; }
    void setRxDelay(float value) { TxRxParametersBuilder::rxDelay = value; }
    void setBitstreamId(const std::optional<BitstreamId> &value) { TxRxParametersBuilder::bitstreamId = value; }
    void setMaskedChannelsTx(const std::unordered_set<ChannelIdx> &value) { TxRxParametersBuilder::maskedChannelsTx = value; }
    void setMaskedChannelsRx(const std::unordered_set<ChannelIdx> &value) { TxRxParametersBuilder::maskedChannelsRx = value; }

private:
    ::std::vector<bool> txAperture;
    ::std::vector<float> txDelays;
    std::optional<::arrus::ops::us4r::Pulse> txPulse;
    ::std::vector<bool> rxAperture;
    Interval<uint32> rxSampleRange;
    int32 rxDecimationFactor;
    float pri;
    Tuple<ChannelIdx> rxPadding;
    float rxDelay;
    std::optional<BitstreamId> bitstreamId;
    std::unordered_set<ChannelIdx> maskedChannelsTx;
    std::unordered_set<ChannelIdx> maskedChannelsRx;
};

class TxRxParametersSequenceBuilder;

class TxRxParametersSequence {
public:
    TxRxParametersSequence() = default;
    TxRxParametersSequence(const std::vector<TxRxParameters> &parameters, const uint16 nRepeats,
                           const std::optional<float> &sri, ops::us4r::TGCCurve tgcCurve,
                           const DeviceId &txProbeId, const DeviceId &rxProbeId)
        : parameters(parameters), nRepeats(nRepeats), sri(sri), tgcCurve(std::move(tgcCurve)), txProbeId(txProbeId),
          rxProbeId(rxProbeId) {}

    [[nodiscard]] std::vector<TxRxParameters> getParameters() const { return parameters; }

    [[nodiscard]] const TxRxParameters &at(size_t i) const { return parameters.at(i); }
    /** Returns the number of ops in the sequence. */
    [[nodiscard]] size_t size() const { return parameters.size(); }
    [[nodiscard]] uint16 getNRepeats() const { return nRepeats; }
    [[nodiscard]] const std::optional<float> &getSri() const { return sri; }
    ops::us4r::TGCCurve getTgcCurve() const { return tgcCurve; }
    const DeviceId &getTxProbeId() const { return txProbeId; }
    const DeviceId &getRxProbeId() const { return rxProbeId; }

    auto begin() const { return std::begin(parameters); }
    auto end() const { return std::end(parameters); }

    /**  Returns the number of actual ops, that is, a the number of ops excluding RxNOPs. */
    [[nodiscard]] uint16 getNumberOfNoRxNOPs() const {
        uint16 res = 0;
        for (const auto &param : parameters) {
            if (!param.isRxNOP()) {
                ++res;
            }
        }
        return res;
    }

    /**
     * Returns a single a unique RX aperture size, or throws IllegalStateException if the size is not unique.
     */
    ChannelIdx getRxApertureSize() const {
        std::unordered_set<ChannelIdx> s;
        for(const auto &p: parameters) {
            if(! p.isRxNOP()) {
                auto padding = p.getRxPadding().sum();
                s.insert(getNumberOfActiveChannels(p.getRxAperture()) + padding);
            }
        }
        ARRUS_REQUIRES_TRUE_IAE(s.size() == 1, "All TX/RXs should have the same RX aperture size "
                                             "and the sequence should not be empty.");
        return *std::begin(s);
    }

    const std::optional<TxRxParameters> getFirstRxOp() const {
        for (auto &op : getParameters()) {
            if (!op.isRxNOP()) {
                return op;
            }
        }
        return std::nullopt;
    }

    void reserve(size_t n) { parameters.reserve(n); }

    const TxRxParameters &getLastOp() const {
        ARRUS_REQUIRES_TRUE(parameters.size() != 0, "Array should not be empty");
        return parameters[parameters.size() - 1];
    }

    friend std::ostream &operator<<(std::ostream &os, const TxRxParametersSequence &sequence) {
        os << "Sequence: ";
        for(const auto &param: sequence.getParameters()) {
            os << param << ", ";
        }
        os << " n repeats: " << sequence.getNRepeats() << ", ";
        os << " SRI: " << arrus::toString(sequence.getSri()) << ", ";
        os << " TGC curve: " << arrus::toString(sequence.getTgcCurve());
        os << std::endl;
        return os;
    }

private:
    friend TxRxParametersSequenceBuilder;
    std::vector<TxRxParameters> parameters;
    uint16 nRepeats{0};
    std::optional<float> sri{0};
    ops::us4r::TGCCurve tgcCurve;
    DeviceId txProbeId{DeviceType::Probe, 0};
    DeviceId rxProbeId{DeviceType::Probe, 0};
};

using TxParametersSequenceColl = std::vector<TxRxParametersSequence>;

class TxRxParametersSequenceBuilder {
public:
    TxRxParametersSequenceBuilder() = default;

    TxRxParametersSequenceBuilder &setCommon(const ops::us4r::TxRxSequence &s) {
        sequence.nRepeats = s.getNRepeats();
        sequence.sri = s.getSri();
        sequence.tgcCurve = s.getTgcCurve();
        sequence.txProbeId = s.getTxProbeId();
        sequence.rxProbeId = s.getRxProbeId();
        return *this;
    }

    TxRxParametersSequenceBuilder &setCommon(const TxRxParametersSequence &s) {
        sequence.nRepeats = s.getNRepeats();
        sequence.sri = s.getSri();
        sequence.tgcCurve = s.getTgcCurve();
        sequence.txProbeId = s.getTxProbeId();
        sequence.rxProbeId = s.getRxProbeId();
        return *this;
    }

    TxRxParametersSequenceBuilder &addEntry(const TxRxParameters &params) {
        sequence.parameters.push_back(params);
        return *this;
    }

    TxRxParametersSequenceBuilder &addEntry(const ops::us4r::TxRx &op) {
        TxRxParametersBuilder builder(op);
        return addEntry(builder.build());
    }

    TxRxParametersSequenceBuilder &resize(size_t n, const TxRxParameters &params) {
        sequence.parameters.resize(n, params);
        return *this;
    }

    TxRxParametersSequence build() {
        auto tmp = std::move(sequence);
        sequence = TxRxParametersSequence{};
        return tmp;
    }

    const TxRxParametersSequence &getCurrent() const { return sequence; }

private:
    TxRxParametersSequence sequence;
};

}// namespace arrus::devices::us4r

#endif//ARRUS_CORE_DEVICES_TXRXPARAMETERS_H
