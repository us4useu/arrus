#ifndef ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H

#include <mutex>
#include <condition_variable>
#include <gsl/span>
#include <chrono>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/framework/FifoBuffer.h"


namespace arrus::devices {

using ::arrus::framework::FifoBuffer;
using ::arrus::framework::FifoBufferElement;

class HostBufferElementImpl: public FifoBufferElement {
public:
    using SharedHandle = std::shared_ptr<HostBufferElementImpl>;

    HostBufferElementImpl(int16 *address, const size_t size)
    : address(address), size(size) {}

    void registerReleaseFunction(std::function<void()> &func) {
        releaseFunction = func;
    }

    void release() override {
        releaseFunction();
    }

    int16* getAddress() {
        return address;
    }


private:
    int16* address;
    size_t size;
    std::function<void()> releaseFunction;
};

/**
 * Us4R system's output circular FIFO buffer.
 *
 * The buffer has the following relationships:
 * - buffer contains **elements**
 * - the **element** is filled by many us4oems (with given ordinal)
 *
 * A single element is the output of a single data transfer (the result of running a complete sequence once).
 *
 * The state of each buffer element is determined by the field accumulators:
 * - accumulators[element] == 0 means that the buffer element was processed and is ready for new data from the producer.
 * - accumulators[element] > 0 && accumulators[element] != filledAccumulator means that the buffer element is partially
 *   confirmed by some of us4oems
 * - accumulators[element] == filledAccumulator means that the buffer element is ready to be processed by a consumer.
 *
 * The assumption is here that each element of the buffer has the same size (and the same us4oem offsets).
 */
class Us4ROutputBuffer: public FifoBuffer {
public:
    static constexpr size_t DATA_ALIGNMENT = 4096;
    using AccumulatorType = uint16;

    /**
     * Buffer's constructor.
     *
     * @param us4oemOutputSizes number of bytes to allocate for each of the
     *  us4oem output. That is, the i-th value describes how many bytes will
     *  be written by i-th us4oem to generate a single buffer element.
     */
    Us4ROutputBuffer(const std::vector<size_t> &us4oemOutputSizes, uint16 nElements,
                     const OnNewDataCallback &callback)
        : elementSize(0),
          accumulators(nElements),
          filledAccumulator((1ul << (size_t) us4oemOutputSizes.size()) - 1) {

        this->initialize();
        ARRUS_REQUIRES_TRUE(us4oemOutputSizes.size() <= 16,
                            "Currently Us4R data buffer supports up to 16 us4oem modules.");
        // Calculate
        size_t us4oemOffset = 0;
        Ordinal us4oemOrdinal = 0;
        for(auto s : us4oemOutputSizes) {
            us4oemOffsets.emplace_back(us4oemOffset);
            us4oemOffset += s;
            if(s == 0) {
                // We should not expect any response from modules, which do not acquire any data.
                filledAccumulator &= ~(1ul << us4oemOrdinal);
            }
            ++us4oemOrdinal;
        }
        elementSize = us4oemOffset;
        // Allocate buffer with an appropriate size.
        dataBuffer = reinterpret_cast<int16*>(operator new[](elementSize*nElements, std::align_val_t(DATA_ALIGNMENT)));

        for(int i = 0; i < nElements; ++i) {
            auto elementAddress = reinterpret_cast<int16*>(reinterpret_cast<int8*>(dataBuffer) + i*elementSize);
            elements.push_back(std::make_shared<HostBufferElementImpl>(elementAddress, elementSize));
        }
        getDefaultLogger()->log(LogSeverity::DEBUG,
                                ::arrus::format("Allocated {} ({}, {}) bytes of memory, address: {}",
                                elementSize*nElements, elementSize, nElements, (size_t)dataBuffer));
    }

    ~Us4ROutputBuffer() override {
        ::operator delete(dataBuffer, std::align_val_t(DATA_ALIGNMENT));
        getDefaultLogger()->log(LogSeverity::DEBUG, "Released the output buffer.");
    }

    [[nodiscard]] uint16 getNumberOfElements() const override {
        return elements.size();
    }

    uint8 *getAddress(uint16 elementNumber, Ordinal us4oem) {
        return reinterpret_cast<uint8*>(this->elements[elementNumber]->getAddress()) + us4oemOffsets[us4oem];
    }

    /**
     * Returns a total size of the buffer, the number of **uint16** values.
     */
    [[nodiscard]] size_t getElementSize() const override {
        return elementSize;
    }

    /**
     * Signals the readiness of new data acquired by the n-th Us4OEM module.
     *
     * This function should be called by us4oem interrupt callbacks.
     *
     * @param n us4oem ordinal number
     *
     *  @return true if the buffer signal was successful, false otherwise (e.g. the queue was shut down).
     */
    bool signal(Ordinal n, uint16 elementNr) {
        std::unique_lock<std::mutex> guard(mutex);
        if(this->state != State::RUNNING) {
            getDefaultLogger()->log(LogSeverity::TRACE, "Signal queue shutdown.");
            return false;
        }
        this->validateState();
        // What if we will be to late in releasing buffer element?
        // 1. if it is a lock-free buffer version - signal error
        // 2. if it is a sync buffer version - wait till the element will be cleared
        // Co gdy nie zdazymy wyczyscic akumulatora? Informacja o nowej ramce nie moze przepasc
        // Dwie opcje:
        // wersja synchroniczna - zablokuj sie i zaczekaj, az element bedzie wolny
        // wersja asynchroniczna - zglos blad i oznacz bufor jako nieprawidlowy
        auto &accumulator = accumulators[elementNr];
        accumulator |= 1ul << n;
        if((accumulator & filledAccumulator) == filledAccumulator) {
            newDataCallback(elements[elementNr]);
        }
        return true;
    }

    /**
     * Releases the front data from further data acquisition.
     *
     * This function should be called by data processing thread when
     * the data is no more needed.
     */
    void release(int elementNr) {
        std::unique_lock<std::mutex> guard(mutex);
        validateState();
        accumulators[elementNr] = 0;
        guard.unlock();
    }

    void markAsInvalid() {
        std::unique_lock<std::mutex> guard(mutex);
        this->state = State::INVALID;
        guard.unlock();
    }

    void shutdown() {
        std::unique_lock<std::mutex> guard(mutex);
        this->state = State::SHUTDOWN;
        guard.unlock();
    }

    void resetState() {
        this->state = State::INVALID;
        this->initialize();
        this->state = State::RUNNING;
    }

    void initialize() {
        accumulators = std::vector<AccumulatorType>(this->getNumberOfElements());
    }

    void registerReleaseFunction(uint16 element, std::function<void()> &releaseFunction) {
        this->elements[element]->registerReleaseFunction(releaseFunction);
    }

private:
    std::mutex mutex;
    /** A size of a single element IN BYTES. */
    size_t elementSize;
    /**  Total size in the number of elements. */
    int16 *dataBuffer;
    /** Host buffer elements */
    std::vector<HostBufferElementImpl::SharedHandle> elements;
    /** Relative addresses where us4oem modules will write. IN NUMBER OF BYTES. */
    std::vector<size_t> us4oemOffsets;

    // element number -> accumulator
    std::vector<AccumulatorType> accumulators;
    /** A pattern of the filled accumulator, which indicates that the
     * whole element is ready. */
    AccumulatorType filledAccumulator;

    OnNewDataCallback newDataCallback;

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
