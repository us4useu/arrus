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
          isAccuClear(nElements),
          heads(n, 0) {
        filledAccumulator = (1ul << n) - 1;
    }

    using AccumulatorType = uint16_t;

    /**
     * Notifies consumers, that new data arrived.
     */
    bool notify(unsigned ordinal) {
        std::unique_lock<std::mutex> guard(mutex);
        if(this->isShutdown) {
            // Nothing to notify about, the connection is closed.
            return false;
        }
        auto &accumulator = accumulators[heads[ordinal]];
        if(accumulator & (1ul << ordinal)) {
            throw std::runtime_error("Tried to overwrite not released buffer.");
        }
        accumulator |= 1ul << ordinal;
        heads[ordinal] = (heads[ordinal] + 1) % nElements;
        bool isElementDone =
            (accumulator & filledAccumulator) == filledAccumulator;
        if(isElementDone) {
            guard.unlock();
            bufferEmpty.notify_one();
        }
        return true;
    }

    /**
     * Reserves access to buffer's head.
     *
     * @return true, if the thread was able to access the element,
     *   false otherwise (e.g. queue shutdown).
     */
    bool reserveElement(Ordinal ordinal) {
        std::unique_lock<std::mutex> guard(mutex);
        auto headIdx = heads[ordinal];
        auto &accumulator = accumulators[headIdx];
        while(accumulator > 0) {
            isAccuClear[headIdx].wait(guard);
            if(this->isShutdown) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns an index to the first element in the buffer that is ready for processing.
     *
     * @timeout number of microseconds to wait till the timeout
     * @return value >= 0 if the thread was able to access tail, -1 if the queue is shutted down, -2 if timeout
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
     * Moves forward the tail and marks the element as ready for further acquisition.
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
        // TODO block the thread till all produces call "reserveElement"
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
    }

    unsigned size() {
        return nElements;
    }

private:
    std::condition_variable bufferEmpty;
    std::mutex mutex;
    std::vector<AccumulatorType> accumulators;
    std::vector<std::condition_variable> isAccuClear;
    AccumulatorType filledAccumulator;
    uint16_t tailIdx{0};
    std::vector<uint16_t> heads;
    unsigned nElements;
    bool isShutdown{false};
};

}

#endif //ARRUS_CORE_DEVICES_US4R_RXBUFFER_H
