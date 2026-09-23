#ifndef ARRUS_CORE_DEVICES_US4R_US4OEMTRANSFERCALLBACKS_H
#define ARRUS_CORE_DEVICES_US4R_US4OEMTRANSFERCALLBACKS_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "arrus/core/api/common/exceptions.h"

namespace arrus::devices {

/**
 * Callbacks for the us4OEM "data transfer (to host) done" interrupts.
 *
 * The us4OEM calls the IRQ callbacks in a round-robin fashion (the i-th PCIDMA interrupt calls the i-th callback),
 * and the list of callbacks cannot be modified while the device is running. This class is registered as the
 * single callback of that list, and dispatches each interrupt to the actual transfer callback, in exactly the
 * same round-robin order.
 *
 * In contrast to the us4OEM callback list, the callbacks can be replaced while the device is running (see
 * the schedule method): this is what the sequencer double-buffering requires (the number of transfers per
 * buffer element may change after swapping the sequencer banks).
 *
 * The callbacks are organized into elements (us4OEM buffer elements): each element consists of exactly
 * nTransfersPerElement transfers (i.e. interrupts); the elements are acquired cyclically.
 */
class Us4OEMTransferCallbacks {
public:
    using SharedHandle = std::shared_ptr<Us4OEMTransferCallbacks>;
    using Callbacks = std::vector<std::function<void()>>;

    /**
     * Sets the callbacks to use. The first interrupt after this call is handled by the first callback of the
     * element 0. Use only when the device is stopped.
     *
     * @param callbacks element -> transfer callback, i.e. nElements*nTransfersPerElement callbacks
     */
    void set(Callbacks callbacks, size_t nTransfersPerElement) {
        std::lock_guard<std::mutex> guard{mutex};
        epoch = Epoch{validate(std::move(callbacks), nTransfersPerElement), nTransfersPerElement, 0};
        processed = 0;
        pending.clear();
    }

    /**
     * Resets the element counter (e.g. on the device start). The next interrupt will be handled by the first
     * callback of the element 0.
     */
    void reset() {
        std::lock_guard<std::mutex> guard{mutex};
        if (!pending.empty()) {
            // The device was stopped (and is started again) before the scheduled callbacks were used; the sequencer
            // is already programmed for the last of them, so use it from the beginning.
            epoch = std::move(pending.back());
            pending.clear();
        }
        epoch.startElement = 0;
        processed = 0;
    }

    /**
     * Schedules new callbacks: they will be used starting from the given element (counted from the last call to
     * set or reset, i.e. the number of elements acquired so far with the previous callbacks). All the elements
     * acquired before will be handled by the current (or previously scheduled) callbacks.
     *
     * Can be called while the device is running, also when the previously scheduled callbacks are not in use yet
     * (fromElement must not decrease).
     *
     * @param callbacks element -> transfer callback, i.e. nElements*nTransfersPerElement callbacks; the element
     *   i of the callbacks will handle the element number fromElement+i (mod nElements).
     */
    void schedule(Callbacks callbacks, size_t nTransfersPerElement, uint64_t fromElement) {
        std::lock_guard<std::mutex> guard{mutex};
        if (!pending.empty() && fromElement < pending.back().startElement) {
            throw IllegalArgumentException("The callbacks must be scheduled in the order of the elements.");
        }
        pending.push_back(Epoch{validate(std::move(callbacks), nTransfersPerElement), nTransfersPerElement, fromElement});
        adoptPendingIfReady();
    }

    /** Returns true if the callbacks provided in the last call to schedule are already in use. */
    bool isScheduledInUse() const {
        std::lock_guard<std::mutex> guard{mutex};
        return pending.empty();
    }

    /**
     * Returns the number of the elements, which were completely handled (i.e. all their transfers are done).
     * The maximum value, when no transfer interrupts are expected (no callbacks).
     */
    uint64_t getNumberOfCompletedElements() const {
        std::lock_guard<std::mutex> guard{mutex};
        if (epoch.callbacks.empty()) {
            return pending.empty() ? std::numeric_limits<uint64_t>::max() : pending.front().startElement;
        }
        return getCurrentElement();
    }

    /** Handles a single "transfer done" interrupt. */
    void operator()() {
        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> guard{mutex};
            adoptPendingIfReady();
            if (epoch.callbacks.empty()) {
                return;
            }
            const auto nPerElement = epoch.nTransfersPerElement;
            const auto nElements = epoch.callbacks.size() / nPerElement;
            const auto element = (epoch.startElement + processed / nPerElement) % nElements;
            callback = epoch.callbacks.at(element * nPerElement + processed % nPerElement);
            ++processed;
        }
        if (callback) {
            callback();
        }
    }

private:
    struct Epoch {
        Callbacks callbacks;
        size_t nTransfersPerElement{0};
        /** The (global) number of the element, for which the first callback should be used. */
        uint64_t startElement{0};
    };

    static Callbacks validate(Callbacks callbacks, size_t nTransfersPerElement) {
        if (callbacks.empty()) {
            return callbacks;
        }
        if (nTransfersPerElement == 0 || callbacks.size() % nTransfersPerElement != 0) {
            throw IllegalArgumentException("The number of transfer callbacks should be a multiple of the number of "
                                           "transfers per element.");
        }
        return callbacks;
    }

    /** The number of the element the next interrupt belongs to (non-empty epoch only). */
    uint64_t getCurrentElement() const {
        return epoch.startElement + processed / epoch.nTransfersPerElement;
    }

    void adoptPendingIfReady() {
        while (!pending.empty()) {
            // No transfers so far: there will be no more interrupts for the current callbacks.
            const bool isEmpty = epoch.callbacks.empty();
            const bool isElementBoundary = !isEmpty && processed % epoch.nTransfersPerElement == 0;
            if (!(isEmpty || (isElementBoundary && getCurrentElement() >= pending.front().startElement))) {
                return;
            }
            epoch = std::move(pending.front());
            pending.pop_front();
            processed = 0;
        }
    }

    mutable std::mutex mutex;
    Epoch epoch;
    /** The number of interrupts handled in the current epoch. */
    uint64_t processed{0};
    /** The scheduled callbacks, in the order of their first element. */
    std::deque<Epoch> pending;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4OEMTRANSFERCALLBACKS_H
