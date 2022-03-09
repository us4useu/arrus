#ifndef ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H

#include <mutex>
#include <condition_variable>
#include <gsl/span>
#include <chrono>
#include <iostream>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/framework/DataBuffer.h"


namespace arrus::devices {

using ::arrus::framework::Buffer;
using ::arrus::framework::BufferElement;

class Us4ROutputBuffer;

/**
 * Buffer element owns the data arrrays, which then are returned to user.
 */
class Us4ROutputBufferElement : public BufferElement {
public:
    using AccumulatorType = uint16;
    using SharedHandle = std::shared_ptr<Us4ROutputBufferElement>;

    Us4ROutputBufferElement(int16 *address, size_t size, const framework::NdArray::Shape &elementShape,
                            const framework::NdArray::DataType elementDataType, AccumulatorType filledAccumulator,
                            size_t position)
        : data(address, elementShape, elementDataType, DeviceId(DeviceType::Us4R, 0)), size(size),
          filledAccumulator(filledAccumulator), position(position)
          {}

    void release() override {
        std::unique_lock<std::mutex> guard(mutex);
        this->accumulator = 0;
        releaseFunction();
    }

    int16 *getAddress() {
        validateState();
        return data.get<int16>();
    }

    framework::NdArray &getData() override {
        validateState();
        return data;
    }

    size_t getSize() override {
        return size;
    }

    size_t getPosition() override {
        return position;
    }

    void registerReleaseFunction(std::function<void()> &func) {
        releaseFunction = func;
    }

    [[nodiscard]] bool isElementReady() {
        std::unique_lock<std::mutex> guard(mutex);
        return accumulator == filledAccumulator;
    }

    void signal(Ordinal n) {
        std::unique_lock<std::mutex> guard(mutex);
        // TODO increase counter for the given element
        // If the given counter is ready, set the accumulator as below
        AccumulatorType us4oemPattern = 1ul << n;
        if((accumulator & us4oemPattern) != 0) {
            throw IllegalStateException("Detected data overflow, buffer is in invalid state.");
        }
        accumulator |= us4oemPattern;
    }

    void resetState() {
        accumulator = 0;
    }

    void markAsInvalid() {
        this->isInvalid = true;
    }

    void validateState() const {
        if(this->isInvalid) {
            throw ::arrus::IllegalStateException(
                    "The buffer is in invalid state (probably some data transfer overflow happened).");
        }
    }

private:
    std::mutex mutex;
    framework::NdArray data;
    size_t size;
    // How many times given element was signaled by i-th us4OEM.
    std::vector<int> signalCounter;
    // How many times given element should be signaled by i-th us4OEM, to consider it as ready.
    std::vector<int> elementReadyCounters;
    AccumulatorType accumulator;
    /** A pattern of the filled accumulator, which indicates that the hole element is ready. */
    AccumulatorType filledAccumulator;
    std::function<void()> releaseFunction;
    bool isInvalid{false};
    size_t position;
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
class Us4ROutputBuffer : public framework::DataBuffer {
public:
    static constexpr size_t DATA_ALIGNMENT = 4096;
    using DataType = int16;

    /**
     * Buffer's constructor.
     *
     * @param us4oemOutputSizes number of bytes to allocate for each of the
     *  us4oem output. That is, the i-th value describes how many bytes will
     *  be written by i-th us4oem to generate a single buffer element.
     */
    Us4ROutputBuffer(const std::vector<size_t> &us4oemOutputSizes,
                     const framework::NdArray::Shape &elementShape,
                     const framework::NdArray::DataType elementDataType,
                     const unsigned nElements,
                     bool stopOnOverflow)
        : elementSize(0) {
        ARRUS_REQUIRES_TRUE(us4oemOutputSizes.size() <= 16,
                            "Currently Us4R data buffer supports up to 16 us4oem modules.");

        size_t nus4oems = us4oemOutputSizes.size();
        Us4ROutputBufferElement::AccumulatorType filledAccumulator((1ul << nus4oems) - 1);
        // Calculate us4oem write offsets for each buffer element.
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
        dataBuffer = reinterpret_cast<DataType *>(
                operator new[](elementSize*nElements, std::align_val_t(DATA_ALIGNMENT)));
        getDefaultLogger()->log(
                LogSeverity::DEBUG,
                ::arrus::format("Allocated {} ({}, {}) bytes of memory, address: {}", elementSize*nElements,
                                elementSize, nElements, (size_t) dataBuffer));

        for(unsigned i = 0; i < nElements; ++i) {
            auto elementAddress = reinterpret_cast<DataType *>(reinterpret_cast<int8 *>(dataBuffer) + i * elementSize);
            elements.push_back(std::make_shared<Us4ROutputBufferElement>(
                    elementAddress, elementSize, elementShape, elementDataType, filledAccumulator, i));
        }
        this->initialize();
        this->stopOnOverflow = stopOnOverflow;
    }

    ~Us4ROutputBuffer() override {
        ::operator delete(dataBuffer, std::align_val_t(DATA_ALIGNMENT));
        getDefaultLogger()->log(LogSeverity::DEBUG, "Released the output buffer.");
    }

    void registerOnNewDataCallback(framework::OnNewDataCallback &callback) override {
        this->onNewDataCallback = callback;
    }

    [[nodiscard]] const framework::OnNewDataCallback &getOnNewDataCallback() const {
        return this->onNewDataCallback;
    }

    void registerOnOverflowCallback(framework::OnOverflowCallback &callback) override {
        this->onOverflowCallback = callback;
    }

    void registerShutdownCallback(framework::OnShutdownCallback &callback) override {
        this->onShutdownCallback = callback;
    }

    [[nodiscard]] size_t getNumberOfElements() const override {
        return elements.size();
    }

    BufferElement::SharedHandle getElement(size_t i) override {
        return std::static_pointer_cast<BufferElement>(elements[i]);
    }

    uint8 *getAddress(uint16 elementNumber, Ordinal us4oem) {
        return reinterpret_cast<uint8 *>(this->elements[elementNumber]->getAddress()) + us4oemOffsets[us4oem];
    }

    /**
     * Returns a total size of the buffer, the number of bytes.
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
            getDefaultLogger()->log(LogSeverity::DEBUG, "Signal queue shutdown.");
            return false;
        }
        this->validateState();
        auto &element = this->elements[elementNr];
        try {
            element->signal(n);
        } catch(const IllegalArgumentException &e) {
            this->markAsInvalid();
            throw e;
        }
        if(element->isElementReady()) {
            guard.unlock();
            onNewDataCallback(elements[elementNr]);
        } else {
            guard.unlock();
        }
        return true;
    }

    void markAsInvalid() {
        std::unique_lock<std::mutex> guard(mutex);
        if(this->state != State::INVALID) {
            this->state = State::INVALID;
            for(auto &element: elements) {
                element->markAsInvalid();
            }
            this->onOverflowCallback();
        }
    }

    void shutdown() {
        std::unique_lock<std::mutex> guard(mutex);
        this->onShutdownCallback();
        this->state = State::SHUTDOWN;
        guard.unlock();
    }

    void resetState() {
        this->state = State::INVALID;
        this->initialize();
        this->state = State::RUNNING;
    }

    void initialize() {
        for(auto &element: elements) {
            element->resetState();
        }
    }

    void registerReleaseFunction(size_t element, std::function<void()> &releaseFunction) {
        this->elements[element]->registerReleaseFunction(releaseFunction);
    }

    bool isStopOnOverflow() {
        return this->stopOnOverflow;
    }


private:
    std::mutex mutex;
    /** A size of a single element IN number of BYTES. */
    size_t elementSize;
    /**  Total size in the number of elements. */
    int16 *dataBuffer;
    /** Host buffer elements */
    std::vector<Us4ROutputBufferElement::SharedHandle> elements;
    /** Relative addresses where us4oem modules will write. IN NUMBER OF BYTES. */
    std::vector<size_t> us4oemOffsets;
    // Callback that should be called once new data arrive.
    framework::OnNewDataCallback onNewDataCallback;
    framework::OnOverflowCallback onOverflowCallback{[]() {}};
    framework::OnShutdownCallback onShutdownCallback{[]() {}};

    // State management
    enum class State {
        RUNNING, SHUTDOWN, INVALID
    };
    State state{State::RUNNING};
    bool stopOnOverflow{true};

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
        } else if(this->state == State::SHUTDOWN) {
            throw ::arrus::IllegalStateException(
                "The data buffer has been turned off.");
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
