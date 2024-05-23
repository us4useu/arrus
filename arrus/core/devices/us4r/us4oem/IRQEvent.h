#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_IRQEVENT_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_IRQEVENT_H

#include <condition_variable>
#include <mutex>
#include <optional>

#include "arrus/common/format.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus::devices {

/**
 * IRQ event (conditional variable). This class allows to wait on / notify about new IRQs coming from the device.
 * It also has the capability of counting incoming IRQs and raises an exception if some missing/unhandled IRQs
 * occur.
 */
class IRQEvent {
public:
    /**
     * Notifies about new IRQ.
     */
    void notifyOne() {
        std::unique_lock l(irqEventMutex);
        ++irqsRegistered;
        irqEvent.notify_one();
    }

    /**
     * Waits for new IRQ to appear.
     */
    void wait(std::optional<long long> timeout = std::nullopt) {
        std::unique_lock lock(irqEventMutex);
        if (timeout.has_value()) {
            bool isReady = irqEvent.wait_for(lock, std::chrono::milliseconds(timeout.value()), [this]() {
                // Wait until the number of registered interrupts is greater than the number of IRQs already handled.
                // (i.e. there is some new, unhandled interrupt).
                return this->irqsRegistered > this->irqsHandled;
            });
            if (!isReady) {
                throw TimeoutException("Timeout on waiting for trigger to be registered. Is the system still alive?");
            }
        } else {
            // No timeout, wait infinitely.
            irqEvent.wait(lock, [this]() { return this->irqsRegistered > this->irqsHandled; });
        }
        if (this->irqsRegistered != this->irqsHandled + 1) {
            // In the correct scenario, we expect that the number of already handled IRQs is equal to the number of
            // registered IRQs minus 1.
            // If it's not true, it means that we have lost some IRQ -- this is an exception that user should react to.
            throw IllegalStateException("The number of registered IRQs is different than the number of handled IRQs."
                                        " We detected missing IRQs.");
        }
        ++this->irqsHandled;
    }

    void resetCounters() {
        irqsHandled = 0;
        irqsRegistered = 0;
    }

private:
    std::mutex irqEventMutex;
    std::condition_variable irqEvent;
    /** The number of IRQs registered by the interrupt handler */
    size_t irqsHandled;
    /** The number of IRQs handled by the wait method */
    size_t irqsRegistered;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_IRQEVENT_H
