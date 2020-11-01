#ifndef ARRUS_CORE_DEVICES_US4R_RXBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_RXBUFFER_H

#include <mutex>

#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::devices {

class RxBuffer {
public:
    RxBuffer(Ordinal n, uint16_t nElements)
        : nElements(nElements),
          accumulators(nElements, 0),
          isAccuClear(nElements) {

        filledAccumulator = (1ul << n) - 1;
    }

    using AccumulatorType = uint16_t;

    /**
     * Notifies consumers, that new data arrived.
     */
    bool notify(unsigned ordinal, unsigned bufferElement) {
        if(this->isShutdown) {
            // Nothing to notify about, the connection is closed.
            return false;
        }
        std::unique_lock<std::mutex> guard(mutex);
        auto &accumulator = accumulators[bufferElement];
        if(accumulator & (1ul << ordinal)) {
            throw std::runtime_error("Tried to overwrite not released buffer.");
        }
        accumulator |= 1ul << ordinal;
        bool isElementDone =
            (accumulator & filledAccumulator) == filledAccumulator;
        if(isElementDone) {
            guard.unlock();
            bufferEmpty.notify_one();
        }
        return true;
    }

    /**
     * Reserves access to i-th buffer element.
     *
     * @return true, if the thread was able to access the element,
     *   false otherwise (e.g. queue shutdown).
     *   TODO(pjarosik) return status object instead of boolean value
     */
    bool reserveElement(int bufferElement) {
        std::unique_lock<std::mutex> guard(mutex);

        auto &accumulator = accumulators[bufferElement];
        while(accumulator > 0) {
            isAccuClear[bufferElement].wait(guard);
            if(this->isShutdown) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns an index to the first element in the buffer that is ready for processing.
     *
     * @return value >= 0 if the thread was able to access tail, negative number otherwise
     *  (e.g. when the queue is shutted down).
     */
    int16_t tail() {
        std::unique_lock<std::mutex> guard(mutex);
        while(accumulators[tailIdx] != filledAccumulator) {
            bufferEmpty.wait(guard);
            if(this->isShutdown) {
                return -1;

            }
        }
        return tailIdx;
    }

    /**
     * Moves forward of the tail and marks the element as ready for further acquisition.
     */
    bool releaseTail() {
        std::unique_lock<std::mutex> guard(mutex);
        auto releasedIdx = tailIdx;
        while(accumulators[releasedIdx] != filledAccumulator) {
            bufferEmpty.wait(guard);
            if(this->isShutdown) {
                return false;
            }
        }
        accumulators[releasedIdx] = 0;
        tailIdx = (tailIdx + 1) % nElements;
        guard.unlock();
        isAccuClear[releasedIdx].notify_all();
        return true;
    }

    /**
     * The object of this class should not be used anymore after the shutdown
     * was called.
     */
    void shutdown() {
        std::unique_lock<std::mutex> guard(mutex);
        isShutdown = true;
        guard.unlock();

        bufferEmpty.notify_all();
        for(auto &cv : isAccuClear) {
            cv.notify_all();
        }

        std::string msg = "rx buffer shutdown\n";
        std::cout << msg;
    }

private:
    std::condition_variable bufferEmpty;
    std::mutex mutex;
    std::vector<AccumulatorType> accumulators;
    std::vector<std::condition_variable> isAccuClear;
    AccumulatorType filledAccumulator;
    uint16_t tailIdx{0};
    unsigned nElements;
    bool isShutdown{false};
};

}

#endif //ARRUS_CORE_DEVICES_US4R_RXBUFFER_H
