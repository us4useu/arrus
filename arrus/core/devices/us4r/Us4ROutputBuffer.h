#ifndef ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H

#include <mutex>
#include <condition_variable>
#include <gsl/span>
#include <chrono>

#include "arrus/core/api/devices/us4r/HostBuffer.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"


namespace arrus::devices {

/**
 * Us4R system's output circular FIFO buffer.
 *
 * The buffer has the following relationships:
 * - buffer contains **elements**
 * - the **element** is filled by many us4oems (with given ordinal)
 *
 * An example of the element is a single RF frame required to reconstruct
 * a single b-mode image.
 *
 * The state of each buffer element is determined by the field `accumulators:
 * - accumulators[element] == 0 means that the buffer element was processed and is ready for new data from the producer.
 * - accumulators[element] > 0 && accumulators[element] != filledAccumulator means that the buffer element is partially confirmed by some of us4oems
 * - accumulators[element] == filledAccumulator means that the buffer element is ready to be processed by a consumer.
 */
class Us4ROutputBuffer: public HostBuffer {
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
    Us4ROutputBuffer(const std::vector<size_t> &us4oemOutputSizes, uint16 nElements)
        : elementSize(0), nElements(nElements), tailIdx(0),
          currentNumberOfElements(0),
          accumulators(nElements), isAccuClear(nElements),
          us4oemPositions(us4oemOutputSizes.size()),
          filledAccumulator((1ul << (size_t) us4oemOutputSizes.size()) - 1) {

        this->initialize();
        // Buffer allocation.
        ARRUS_REQUIRES_TRUE(us4oemOutputSizes.size() <= 16,
                            "Currently Us4R data buffer supports up to 16 us4oem modules.");
        size_t us4oemOffset = 0;

        Ordinal us4oemOrdinal = 0;
        for(auto s : us4oemOutputSizes) {
            this->us4oemOffsets.emplace_back(us4oemOffset);
            us4oemOffset += s;
            if(s == 0) {
                // We should not expect any response from modules, that do not acquire any data.
                filledAccumulator &= ~(1ul << us4oemOrdinal);
            }
            ++us4oemOrdinal;
        }
        elementSize = us4oemOffset;
        dataBuffer = reinterpret_cast<int16*>(operator new[](elementSize * nElements, std::align_val_t(DATA_ALIGNMENT)));
        getDefaultLogger()->log(LogSeverity::DEBUG, ::arrus::format("Allocated {} ({}, {}) bytes of memory, address: {}",
                                                                    elementSize*nElements, elementSize, nElements, (size_t)dataBuffer));
    }

    ~Us4ROutputBuffer() override {
        ::operator delete(dataBuffer, std::align_val_t(DATA_ALIGNMENT));
        getDefaultLogger()->log(LogSeverity::DEBUG, "Released the output buffer.");
    }

    [[nodiscard]] uint16 getNumberOfElements() const {
        return nElements;
    }

    uint8 *getAddress(uint16 elementNumber, Ordinal us4oem) {
        return reinterpret_cast<uint8*>(dataBuffer) + elementNumber * elementSize + us4oemOffsets[us4oem];
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
     *   accumulator or queue clearance (each separately), 0 means no timeout
     *
     *  @throws TimeoutException when accumulator clearance of push operation
     *   reaches the timeout
     *  @return true if the buffer signal was successful, false otherwise (e.g. the queue was shut down).
     */
    bool signal(Ordinal n, int firing, long long timeout) {
        std::unique_lock<std::mutex> guard(mutex);
        if(this->state != State::RUNNING) {
            getDefaultLogger()->log(LogSeverity::TRACE, "Signal queue shutdown.");
            return false;
        }
        auto &accumulator = accumulators[firing];
        getDefaultLogger()->log(LogSeverity::TRACE,
                                ::arrus::format("Signal, position: {}, accumulator: {}", firing, accumulator));

        while(accumulator & (1ul << n)) {
            // wait till the bit will be cleared
            getDefaultLogger()->log(
                LogSeverity::TRACE,
                arrus::format("Us4OEM:{} signal thread is waiting for accumulator clearance: {}", n, firing));
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                isAccuClear[firing], guard, timeout,
                ::arrus::format("Us4OEM:{} Timeout while waiting for queue element clearance.", n))
            if(this->state != State::RUNNING) {
                getDefaultLogger()->log(LogSeverity::TRACE, "Signal queue shutdown.");
                return false;
            }
        }
        accumulator |= 1ul << n;
        bool isElementReady = (accumulator & filledAccumulator) == filledAccumulator;
        if(isElementReady) {
            guard.unlock();
            queueEmpty.notify_one();
        }
        return true;
    }

    /**
     * This function just waits till the given element will be cleared by a consumer.
     */
    bool waitForRelease(Ordinal n, int firing, long long timeout) {
        std::unique_lock<std::mutex> guard(mutex);
        if(this->state != State::RUNNING) {
            return false;
        }

        auto &accumulator = accumulators[firing];

        while(accumulator != 0) {
            // wait till the bit will be cleared
            getDefaultLogger()->log(
                LogSeverity::TRACE,
                arrus::format("Us4OEM:{} signal thread is waiting for accumulator clearance: {}", n, firing));
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                isAccuClear[firing], guard, timeout,
                ::arrus::format("Us4OEM:{} Timeout while waiting for queue element clearance.", n))
            if(this->state != State::RUNNING) {
                return false;
            }
        }
        return true;
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
    void releaseTail(long long timeout) override {
        std::unique_lock<std::mutex> guard(mutex);
        validateState();
        auto releasedIdx = tailIdx;
        while(accumulators[releasedIdx] != filledAccumulator) {
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                queueEmpty, guard, timeout,
                "Timeout while waiting for new data queue.")
            validateState();
        }
        accumulators[releasedIdx] = 0;
        tailIdx = (tailIdx + 1) % nElements;
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
    short *tail(long long timeout) override {
        std::unique_lock<std::mutex> guard(mutex);
        validateState();
        while(accumulators[tailIdx] != filledAccumulator) {
            ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                queueEmpty, guard, timeout,
                "Timeout while waiting for new data queue.")
            validateState();
        }
        return (int16*)((uint8*)dataBuffer + tailIdx * elementSize);
    }

    void markAsInvalid() {
        // TODO this function should have the "highest priority" possible
        std::unique_lock<std::mutex> guard(mutex);
        this->state = State::INVALID;
        guard.unlock();
        queueEmpty.notify_all();
        for(auto &cv: isAccuClear) {
            cv.notify_all();
        }
    }

    void shutdown() {
        std::unique_lock<std::mutex> guard(mutex);
        this->state = State::SHUTDOWN;
        guard.unlock();
        queueEmpty.notify_all();
        for(auto &cv: isAccuClear) {
            cv.notify_all();
        }
    }

    void resetState() {
        this->state = State::INVALID;
        this->initialize();
        this->state = State::RUNNING;
    }

    void initialize() {
        this->tailIdx = 0;
        accumulators = std::vector<AccumulatorType>(this->nElements);
        isAccuClear = std::vector<std::condition_variable>(this->nElements);
        for(auto &pos : us4oemPositions) {
            pos = 0;
        }
    }

    int16 *head(long long int) override {
        throw ::arrus::ArrusException("Not implemented.");
    }

private:
    size_t elementSize;
    /** Us4OEM output address relative to the data buffer element address. */
    std::vector<size_t> us4oemOffsets;
    /**  Total size in the number of elements. */
    uint16 nElements;
    int16 *dataBuffer;
    uint16 tailIdx;
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

    // State management
    enum class State {RUNNING, SHUTDOWN, INVALID};
    State state{State::RUNNING};

    /**
     * Throws IllegalStateException when the buffer is in invalid state.
     *
     * @return true if the queue execution should continue, false otherwise.
     */
    void validateState() {
        if(this->state == State::INVALID) {
            throw ::arrus::IllegalStateException(
                "The buffer is in invalid state "
                "(probably some data transfer overflow happened).");
        }
        else if(this->state == State::SHUTDOWN) {
            throw ::arrus::IllegalStateException(
                "The data buffer has been turned off.");
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
