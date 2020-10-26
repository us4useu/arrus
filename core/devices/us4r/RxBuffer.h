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

    void notify(unsigned ordinal, unsigned bufferElement) {
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
    }

    void reserveElement(int bufferElement) {
        std::unique_lock<std::mutex> guard(mutex);
        auto &accumulator = accumulators[bufferElement];
        while(accumulator > 0) {
            isAccuClear[bufferElement].wait(guard);
        }
    }

    uint16_t tail() {
        std::unique_lock<std::mutex> guard(mutex);
        while(accumulators[headIdx] != filledAccumulator) {
            bufferEmpty.wait(guard);
        }
        return headIdx;
    }

    void releaseTail() {
        std::unique_lock<std::mutex> guard(mutex);
        auto releasedIdx = headIdx;
        while(accumulators[releasedIdx] != filledAccumulator) {
            bufferEmpty.wait(guard);
        }
        accumulators[releasedIdx] = 0;
        headIdx = (headIdx + 1) % nElements;
        guard.unlock();
        isAccuClear[releasedIdx].notify_all();
    }

private:
    std::condition_variable bufferEmpty;
    std::mutex mutex;
    std::vector<AccumulatorType> accumulators;
    std::vector<std::condition_variable> isAccuClear;
    AccumulatorType filledAccumulator;
    uint16_t headIdx{0};
    unsigned nElements;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_RXBUFFER_H
