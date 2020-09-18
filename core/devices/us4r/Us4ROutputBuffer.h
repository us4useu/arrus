#ifndef ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H

#include <mutex>
#include <condition_variable>
#include <gsl/span>
#include <chrono>

#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"


namespace arrus::devices {

/**
 * Us4R system's output circular FIFO buffer.
 *
 * The buffer represents following relationships:
 * - buffer contains **elements**
 * - the **element** is filled by many us4oems (with given ordinal)
 */
class Us4ROutputBuffer {
public:
    static constexpr size_t DATA_ALIGNMENT = 4096;
    using AccumulatorType = uint16;

    /**
     * Buffer's constructor.
     *
     * @param us4oemOutputSizes number of samples to allocate for each of the
     *  us4oem output. That is, the i-th element describes how many samples will
     *  be written by i-th us4oem.
     */
    Us4ROutputBuffer(gsl::span<size_t> us4oemOutputSizes, uint16 nElements)
        : elementSize(0), nElements(nElements), headIdx(0),
          currentNumberOfElements(0),
          accumulators(nElements), isAccuClear(nElements),
          us4oemPositions(us4oemOutputSizes.size()),
          filledAccumulator((1ul << (size_t) us4oemOutputSizes.size()) - 1) {

        // Buffer allocation.
        ARRUS_REQUIRES_TRUE(us4oemOutputSizes.size() <= 16,
                            "Currently Us4R buffer supports up "
                            "to 16 us4oem modules.");
        size_t us4oemOffset = 0;
        for(auto s : us4oemOutputSizes) {
            ARRUS_REQUIRES_TRUE(
                s % 64 == 0,
                format("Each element of buffer should have number of "
                       "samples divisible by 64 (actual: {}", s));
            this->us4oemOffsets.emplace_back(us4oemOffset);
            // Each 's' is in the number of samples per line.
            us4oemOffset += s * Us4OEMImpl::N_RX_CHANNELS;
        }
        // element size in number of uint16 elements
        elementSize = us4oemOffset;
        dataBuffer = static_cast<uint16 *>(
            operator new[](
                elementSize * nElements * sizeof(Us4OEMImpl::OutputDType),
                std::align_val_t(DATA_ALIGNMENT)));
    }

    virtual ~Us4ROutputBuffer() {
        ::operator delete(dataBuffer, std::align_val_t(DATA_ALIGNMENT));
        getDefaultLogger()->log(LogSeverity::DEBUG, "Released the output buffer.");
    }

    uint16 *getAddress(uint16 elementNumber, Ordinal us4oem) {
        return dataBuffer + elementNumber * elementSize
               + us4oemOffsets[us4oem];
    }

    /**
     * Returns a total size of the buffer, the number of **uint16** values.
     */
    [[nodiscard]] size_t getElementSize() const {
        return elementSize;
    }

    /**
     * Signals the readiness of new data acquired by the n-th Us4OEM module.
     *
     * This function should be called by us4oem interrupt callbacks.
     *
     * @param n us4oem ordinal number
     * @param timeout number of milliseconds the thread will wait for
     *   accumulator or queue clearance (each separately), nullptr means not
     *   timeout
     *
     *  @throws TimeoutException when accumulator clearance of push operation
     *   reaches the timeout
     */
    void signal(Ordinal n,
                std::optional<unsigned> timeout = 10000,
                const std::optional<std::function<void()>> &function = nullptr) {
        std::unique_lock<std::mutex> guard(mutex);

        auto &us4oemPosition = us4oemPositions[n];
        auto &accumulator = accumulators[us4oemPosition];

        getDefaultLogger()->log(LogSeverity::TRACE,
                                ::arrus::format(
                                    "Signal, position: {}, accumulator: {}",
                                    us4oemPosition, accumulator));

        while(accumulator & (1ul << n)) {
            // wait till the bit will be cleared
            getDefaultLogger()->log(
                LogSeverity::TRACE,
                arrus::format("Us4OEM:{} signal thread is "
                              "waiting for accumulator clearance: {}", n,
                              us4oemPosition));
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                isAccuClear[us4oemPosition], guard, timeout,
                ::arrus::format(
                    "Us4OEM: {} Timeout while waiting for queue element clearance.",
                    n))
        }

        if(function.has_value()) {
            function.value()();
        }
        accumulator |= 1ul << n;

        bool isElementDone =
            (accumulator & filledAccumulator) == filledAccumulator;
        us4oemPosition = (us4oemPosition + 1) % nElements;

        if(isElementDone) {
            guard.unlock();
            queueEmpty.notify_one();
        }
    }

    /**
     * Releases the front data from further data acquisition.
     *
     * This function should be called by data processing thread when
     * the data is no more needed.
     *
     * @param timeout a number of milliseconds the thread will wait when
     * the queue is empty; nullptr means no timeout.
     */
    void releaseFront(std::optional<unsigned> timeout = 100000) {
        std::unique_lock<std::mutex> guard(mutex);
        auto releasedIdx = headIdx;

        while(accumulators[releasedIdx] != filledAccumulator) {
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                queueEmpty, guard, timeout,
                "Timeout while waiting for new data queue.")
        }
        accumulators[releasedIdx] = 0;
        headIdx = (headIdx + 1) % nElements;
        guard.unlock();
        isAccuClear[releasedIdx].notify_all();
    }

    /**
     * Returns a pointer to the front element of the buffer.
     *
     * The method should be called by data processing thread.
     *
     * @param timeout a number of milliseconds the thread will wait when
     * the input queue is empty; nullptr means no timeout.
     * @return a pointer to the front of the queue
     */
    unsigned short *front(std::optional<unsigned> timeout = 100000) {
        std::unique_lock<std::mutex> guard(mutex);
        while(accumulators[headIdx] != filledAccumulator) {
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                queueEmpty, guard, timeout,
                "Timeout while waiting for new data queue.")
        }
        return dataBuffer + headIdx * elementSize;
    }

private:
    /** The number of uint16 values that each element of the buffer contains.
     * A single element consists of all the frame data collected from
     * all us4oem modules. */
    size_t elementSize;
    /** Us4OEM output address relative to the data buffer element address. */
    std::vector<size_t> us4oemOffsets;
    /**  Total size in the number of elements. */
    uint16 nElements;
    uint16 *dataBuffer;
    uint16 headIdx;
    /** Currently occupied size of the buffer. */
    uint16 currentNumberOfElements;

    std::condition_variable queueEmpty;

    std::mutex mutex;
    // frame number -> accumulator
    std::vector<AccumulatorType> accumulators;
    /** A pattern of the filled accumulator, which indicates that the
     * whole element is ready. */
    AccumulatorType filledAccumulator;
    // frame number -> condition variable to notify, that accu is clear
    std::vector<std::condition_variable> isAccuClear;
    // us4oem module id -> current writing position for this us4oem
    std::vector<int> us4oemPositions;


};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
