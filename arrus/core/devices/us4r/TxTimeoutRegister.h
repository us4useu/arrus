#ifndef ARRUS_CORE_DEVICES_US4R_TXTIMEOUTREGISTER_H
#define ARRUS_CORE_DEVICES_US4R_TXTIMEOUTREGISTER_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <optional>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/common/hash.h"
#include "arrus/common/format.h"
#include "arrus/common/utils.h"

namespace arrus::devices {

class TxTimeoutRegisterFactory;

/**
 * A data object that stores all the information regarding TX timeouts to be set on the us4OEMs.
 */
class TxTimeoutRegister {
public:
    using FiringAddress = std::pair<SequenceId, OpId>;

    bool empty() const {
        return timeoutUs.empty();
    }

    /**
     * Returns a collections of TX timeouts to be set in the us4OEM sequences.
     * The id of the tx timeout is equal to the position of the timeout in the output vector.
     *
     * @return timeout in microseconds
     */
    const std::vector<TxTimeout> &getTimeouts() const {
        return timeoutUs;
    }

    TxTimeoutId getTimeoutId(const FiringAddress &address) const {
        return timeoutIds.at(address);
    }

private:
    friend TxTimeoutRegisterFactory;

    TxTimeoutRegister() {}
    std::vector<TxTimeout> timeoutUs;
    std::unordered_map<FiringAddress, TxTimeoutId, PairHash<SequenceId, OpId>> timeoutIds;
};

/**
 * TX time value factory.
 *
 * This class determines the TX interrupt values based on the TX pulse length (TX op) or RX delay (TX NOP).
 * By TX NOP we mean the tx with no active elements in the TX aperture.
 *
 * The output timeouts are in the ascending order.
 */
class TxTimeoutRegisterFactory {
public:
    static constexpr uint16_t EPSILON = 10; // an additional margin for TX timeout [us]

    explicit TxTimeoutRegisterFactory(
        size_t nTimeouts,
        std::function<float(float)> actualTxFunc,
        const std::vector<std::vector<float>> &rxDelays
        ): nTimeouts(nTimeouts), actualTxFunc(std::move(actualTxFunc)), rxDelays(rxDelays) {}

    TxTimeoutRegister createFor(const std::vector<::arrus::ops::us4r::TxRxSequence> &sequences)  {
        SequenceId sId = 0;
        OpId opId = 0;

        if(nTimeouts == 0) {
            // empty register
            return TxTimeoutRegister{};
        }
        // Determine buckets
        std::vector<TxTimeout> timeouts; // [us]
        std::vector<TxTimeout> waveformTimes; // [us]

        sId = 0;
        for(const auto &s: sequences) {
            opId = 0;
            for(const auto &op: s.getOps()) {
                const auto waveformTime = getWaveformTime(op, rxDelays.at(sId).at(opId));
                waveformTimes.push_back(waveformTime);
                ++opId;
            }
            ++sId;
        }

        std::sort(std::begin(waveformTimes), std::end(waveformTimes), std::greater{}); // descending order
        TxTimeout timeout = waveformTimes.at(0);
        timeouts.push_back(timeout + EPSILON);
        for(auto t: waveformTimes) {
            auto prevTimeout = timeout;
            auto it = timeout;
            while(it > t) {
                prevTimeout = it;
                it = it / 2; // NOTE arbitrary heuristic
            }
            if(prevTimeout != timeout) {
                timeouts.push_back(prevTimeout + EPSILON);
                timeout = prevTimeout;
            }
            if(timeouts.size() == nTimeouts) {
                break;
            }
        }
        // Timeouts are in the descending order
        std::reverse(std::begin(timeouts), std::end(timeouts)); // now, they're in the ascending order
        // assign TX/RXs to buckets

        TxTimeoutRegister result;
        result.timeoutUs = timeouts;

        // Assign to buckets.
        sId = 0;
        for(const auto &s: sequences) {
            opId = 0;
            for(const auto &op: s.getOps()) {
                TxTimeout txTime = getWaveformTime(op, rxDelays.at(sId).at(opId)) + EPSILON;
                // Find the first timeout, that is greater or equal than the given tx time.
                auto it = std::find_if(std::begin(timeouts), std::end(timeouts),
                             [txTime](auto t) {return t >= txTime; });
                if(it == std::end(timeouts)) {
                    throw std::runtime_error(format("Couldn't find a timeout that would be greater "
                                                    "or equal than tx time: {} [us]", txTime));
                }
                auto timeoutId = ARRUS_SAFE_CAST(std::distance(std::begin(timeouts), it), uint8_t);
                result.timeoutIds.insert({{sId, opId}, timeoutId});

                ++opId;
            }
            ++sId;
        }
        return result;
    }


private:
    static constexpr TxTimeout MAX_TIMEOUT = (1 << 11) - 1; // TODO move that to the Us4OEMDescriptor?

    /**
     * Returns TX time when the given op has non-empty TX aperture, or RX delay in case we have TX NOP.
     */
    TxTimeout getWaveformTime(const ops::us4r::TxRx &op, float rxDelay) const {
        if(op.getTx().isNOP()) {
            return TxTimeout (std::ceil(rxDelay*1e6f));
        }
        else {
            TxTimeout txTime = getTxTimeUs(op);
            if(txTime > MAX_TIMEOUT) {
                throw IllegalArgumentException(
                    format("TX time {} is higher than the maximum timeout: {}", txTime, MAX_TIMEOUT));
            }
            return txTime;
        }
    }

    [[nodiscard]] TxTimeout getTxTimeUs(const ops::us4r::TxRx &op) const {
        const auto &delays = op.getTx().getDelaysApertureOnly();
        float maxDelay = *std::max_element(std::begin(delays), std::end(delays));
        float frequency = actualTxFunc(op.getTx().getExcitation().getCenterFrequency());
        float nPeriods = op.getTx().getExcitation().getNPeriods();
        float burstTime = 1.0f/frequency*nPeriods;
        auto txTimeUs = ARRUS_SAFE_CAST(std::roundf((maxDelay + burstTime)*1e6f), TxTimeout);
        return txTimeUs;
    }

    size_t nTimeouts{0};
    std::function<float(float)> actualTxFunc;
    std::vector<std::vector<float>> rxDelays;
};

}

#endif//ARRUS_CORE_DEVICES_US4R_TXTIMEOUTREGISTER_H
