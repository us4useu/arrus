#ifndef ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H

#include <chrono>
#include <condition_variable>
#include <format>
#include <iostream>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#undef ERROR // conflicts with LogSeverity::ERROR
#else
#include <sys/mman.h>
#endif

#include "arrus/common/asserts.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/framework/DataBuffer.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/gpu/CudaRuntime.h"
#include "us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

using ::arrus::framework::Buffer;
using ::arrus::framework::BufferElement;

class Us4ROutputBuffer;

/**
 * This class defines the layout of each output array.
 */
class Us4ROutputBufferArrayDef {
public:
    Us4ROutputBufferArrayDef(framework::NdArrayDef definition, size_t address, std::vector<size_t> oemSizes)
        : definition(std::move(definition)), address(address), oemSizes(std::move(oemSizes)) {
        size_t oemAddress = 0;
        for (const auto size : this->oemSizes) {
            oemAddresses.push_back(oemAddress);
            oemAddress += size;
        }
    }

    size_t getAddress() const { return address; }
    const framework::NdArrayDef &getDefinition() const { return definition; }
    size_t getSize() const { return definition.getSize(); }
    /*** Returns address of data produced by the given OEM, relative to the beginning of the element. */
    size_t getOEMAddress(Ordinal oem) const { return address + oemAddresses[oem]; }
    /** Returns the size of this array data produced by the given OEM */
    size_t getOEMSize(Ordinal oem) const {
        ARRUS_REQUIRES_TRUE(oem < oemSizes.size(), "OEM outside of range");
        return oemSizes.at(oem);
    }

private:
    framework::NdArrayDef definition;
    /** Array address, relative to the beginning of the parent element */
    size_t address;
    std::vector<size_t> oemSizes;
    /** The part of array the given OEM, relative to the beginning of the array. */
    std::vector<size_t> oemAddresses;
};

/**
 * Buffer element owns the data arrrays, which then are returned to user.
 */
class Us4ROutputBufferElement : public BufferElement {
public:
    using Accumulator = uint16;
    using SharedHandle = std::shared_ptr<Us4ROutputBufferElement>;

    Us4ROutputBufferElement(size_t position, Tuple<framework::NdArray> arrays, Accumulator filledAccumulator)
        : position(position), arrays(arrays), filledAccumulator(filledAccumulator) {
        const auto &arr = arrays.getValues();
        size = std::accumulate(std::begin(arr), std::end(arr), size_t(0),
                               [](const auto &s, const framework::NdArray &b) { return s + b.nbytes(); });
    }

    void release() override {
        std::unique_lock<std::mutex> guard(mutex);
        this->accumulator = 0;
        releaseFunction();
        this->state = State::FREE;
    }

    /**
     * Calls receive release function (e.g. clears RX flags in the sequencer table).
     */
    void releaseReceive() { receiveReleaseFunction(); }

    int16 *getAddress(ArrayId id) {
        validateState();
        return arrays.getMutable(id).get<int16>();
    }

    /** TODO Deprecated, use getAddress(arrayId) */
    int16 *getAddress() { return getAddress(0); }

    /**
     * This method allows to read element's address regardless of it's state.
     * This method can be used e.g. in a clean-up procedures, that may
     * be called even after some buffer overflow.
     * TODO deprecated, use getAddressUnsafe(arrayId)
     */
    int16 *getAddressUnsafe(ArrayId id) { return arrays.getMutable(id).get<int16>(); }

    int16 *getAddressUnsafe() { return getAddressUnsafe(0); }

    framework::NdArray &getData(ArrayId id) override {
        validateState();
        return arrays.getMutable(id);
    }

    framework::NdArray &getData() override { return getData(0); }

    /**
     * Returns the size of the given element in bytes (equal to the size of all sub-arrays).
     */
    size_t getSize() override { return size; }

    size_t getPosition() override { return position; }

    void registerReleaseFunction(std::function<void()> &f) { releaseFunction = f; }
    void registerReceiveReleaseFunction(std::function<void()> &f) { receiveReleaseFunction = f; }

    [[nodiscard]] bool isElementReady() {
        std::unique_lock<std::mutex> guard(mutex);
        return state == State::READY;
    }

    void signal(Ordinal n) {
        std::unique_lock<std::mutex> guard(mutex);
        Accumulator us4oemPattern = 1ul << n;
        if ((accumulator & us4oemPattern) != 0) {
            throw IllegalStateException("Detected data overflow, buffer is in invalid state.");
        }
        accumulator |= us4oemPattern;
        if (accumulator == filledAccumulator) {
            this->state = State::READY;
        }
    }

    void resetState() {
        accumulator = 0;
        this->state = State::FREE;
    }

    void markAsInvalid() { this->state = State::INVALID; }

    void validateState() const {
        if (getState() == State::INVALID) {
            throw IllegalStateException(
                "The buffer is in invalid state (probably some data transfer overflow happened).");
        }
    }

    [[nodiscard]] State getState() const override { return this->state; }

    uint16 getNumberOfArrays() const override { return ARRUS_SAFE_CAST(arrays.size(), ArrayId); }

private:
    std::mutex mutex;
    size_t position;
    Tuple<framework::NdArray> arrays;
    /** A pattern of the filled accumulator, which indicates that the hole element is ready. */
    Accumulator filledAccumulator;
    /** Size of the whole element (i.e. the sum of all arrays). */
    size_t size;
    // How many times given element was signaled by i-th us4OEM.
    std::vector<int> signalCounter;
    // How many times given element should be signaled by i-th us4OEM, to consider it as ready.
    std::vector<int> elementReadyCounters;
    Accumulator accumulator;
    std::function<void()> releaseFunction;
    /** The function that will be called when the data reception (to the us4OEMs DDR memory) already has ended. */
    std::function<void()> receiveReleaseFunction;
    State state{State::FREE};
};

/**
 * Us4R system's output circular FIFO buffer.
 *
 * The buffer has the following relationships:
 * - buffer contains **elements**
 * - the **element** is filled by many us4oems
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
    using Handle = std::unique_ptr<Us4ROutputBuffer>;
    using SharedHandle = std::shared_ptr<Us4ROutputBuffer>;
    static constexpr size_t ALIGNMENT = 4096;
    static constexpr framework::NdArrayDef::DataType ARRAY_DATA_TYPE = framework::NdArrayDef::DataType::INT16;
    using DataType = int16;
    using Accumulator = Us4ROutputBufferElement::Accumulator;
    using Elements = std::vector<Us4ROutputBufferElement::SharedHandle>;

    /**
     * Buffer's constructor.
     *
     * @param noems: the total number of OEMs, regardless of whether that OEM produces data or not
     *
     */
    Us4ROutputBuffer(Tuple<Us4ROutputBufferArrayDef> arrays, const unsigned nElements, bool stopOnOverflow,
                     size_t noems, bool useP2pDma)
        : elementSize(0), stopOnOverflow(stopOnOverflow), arrayDefs(std::move(arrays)), useP2pDma(useP2pDma) {

        ARRUS_REQUIRES_TRUE(noems <= 16, "Currently Us4R data buffer supports up to 16 OEMs.");

        Accumulator elementReadyPattern = createElementReadyPattern(arrayDefs, noems);
        elementSize = calculateElementSize(arrayDefs);
        try {
            dataBufferSize = elementSize * nElements;
            getDefaultLogger()->log(
                LogSeverity::DEBUG,
                std::format("Allocating {} ({}, {}) bytes of memory, useP2pDma={}", dataBufferSize, elementSize, nElements, useP2pDma));

            if (useP2pDma) {
                auto &cuda = CudaRuntime::instance();
                if (!cuda.isAvailable()) {
                    throw std::runtime_error(
                        "P2P DMA was requested, but the CUDA Runtime library could not be loaded");
                }
                // Detect if we're dealing with an IGPU with unified memory, which does not require special handling for P2P DMA.
                // For now assume that devices with more than 1 GPU do not use unified memory
                int deviceCount = cuda.getDeviceCount();
                if (deviceCount == 0) {
                    throw std::runtime_error("No CUDA devices found");
                }
                // NOTE: integrated seems to return 48 (???) on RTX 5000 (???), so check unifiedAddressing too.
                bool integrated = cuda.getDeviceIntegrated(0);
                bool unifiedAddressing = cuda.getDeviceUnifiedAddressing(0);
                usesUnifiedMemory = deviceCount == 1 && integrated && unifiedAddressing;

                getDefaultLogger()->log(
                    LogSeverity::DEBUG,
                    std::format("CUDA device 0, integrated: {}, unifiedAddressing: {}, unified memory: {}",
                           integrated, unifiedAddressing, usesUnifiedMemory));

                if (!usesUnifiedMemory) {
                    dataBuffer = reinterpret_cast<DataType *>(cuda.malloc(dataBufferSize));
                    if (dataBuffer == nullptr) {
                        getDefaultLogger()->log(LogSeverity::ERROR, "cudaMalloc failed");
                        throw std::runtime_error("cudaMalloc failed");
                    }
                } else {
                    // For integrated GPUs with unified memory, we use cudaHostAlloc
                    dataBuffer = reinterpret_cast<DataType *>(cuda.hostAllocDefault(dataBufferSize));
                    if (dataBuffer == nullptr) {
                        getDefaultLogger()->log(LogSeverity::ERROR, "cudaHostAlloc failed");
                        throw std::runtime_error("cudaHostAlloc failed");
                    }
                }
            } else {
                dataBuffer = reinterpret_cast<DataType *>(mallocChunked(dataBufferSize, ALLOC_CHUNK_SIZE));
            }
            getDefaultLogger()->log(LogSeverity::DEBUG, std::format("Allocated address: {}", (size_t) dataBuffer));
            createElements(arrayDefs, elementReadyPattern, nElements, elementSize);
        } catch (...) {
            releaseDataBuffer();
            dataBuffer = nullptr;
            getDefaultLogger()->log(LogSeverity::DEBUG, "Released the output buffer.");
            throw;
        }
        this->initialize();
    }

    ~Us4ROutputBuffer() override {
        releaseDataBuffer();
        getDefaultLogger()->log(LogSeverity::DEBUG, "Released the output buffer.");
    }

    void registerOnNewDataCallback(framework::OnNewDataCallback &callback) override {
        this->onNewDataCallback = callback;
    }

    [[nodiscard]] const framework::OnNewDataCallback &getOnNewDataCallback() const { return this->onNewDataCallback; }

    void registerOnOverflowCallback(framework::OnOverflowCallback &callback) override {
        this->onOverflowCallback = callback;
    }

    void registerShutdownCallback(framework::OnShutdownCallback &callback) override {
        this->onShutdownCallback = callback;
    }

    [[nodiscard]] size_t getNumberOfElements() const override { return elements.size(); }

    BufferElement::SharedHandle getElement(size_t i) override {
        return std::static_pointer_cast<BufferElement>(elements[i]);
    }

    /**
     * Return address (beginning) of the given buffer element.
     */
    uint8 *getAddress(uint16 bufferElementId) {
        return reinterpret_cast<uint8 *>(this->elements[bufferElementId]->getAddress());
    }

    uint8 *getAddressUnsafe(uint16 elementNumber) {
        return reinterpret_cast<uint8 *>(this->elements[elementNumber]->getAddressUnsafe());
    }

    /**
     * Returns a total size of the buffer, the number of bytes.
     */
    [[nodiscard]] size_t getElementSize() const override { return elementSize; }

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
        if (this->state != State::RUNNING) {
            getDefaultLogger()->log(LogSeverity::DEBUG, "Signal queue shutdown.");
            return false;
        }
        this->validateState();
        auto &element = this->elements[elementNr];
        try {
            element->signal(n);
        } catch (const IllegalArgumentException &e) {
            this->markAsInvalid();
            throw e;
        }
        if (element->isElementReady()) {
            guard.unlock();
            elements[elementNr]->releaseReceive();
            onNewDataCallback(elements[elementNr]);
        } else {
            guard.unlock();
        }
        return true;
    }

    void markAsInvalid() {
        std::unique_lock<std::mutex> guard(mutex);
        if (this->state != State::INVALID) {
            this->state = State::INVALID;
            for (auto &element : elements) {
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
        for (auto &element : elements) {
            element->resetState();
        }
    }

    void registerReleaseFunction(size_t element, std::function<void()> &releaseFunction) {
        this->elements[element]->registerReleaseFunction(releaseFunction);
    }

    void registerReceiveReleaseFunction(size_t element, std::function<void()> &releaseFunction) {
        this->elements[element]->registerReceiveReleaseFunction(releaseFunction);
    }

    bool isStopOnOverflow() { return this->stopOnOverflow; }

    size_t getNumberOfElementsInState(BufferElement::State s) const override {
        size_t result = 0;
        for (size_t i = 0; i < getNumberOfElements(); ++i) {
            if (elements[i]->getState() == s) {
                ++result;
            }
        }
        return result;
    }
    /**
     * Returns relative address of the element area dedicated for the given array, given OEM.
     * The address is relative to the beginning of the whole element (i.e. array 0, oem 0, where
     * 0 is the first non-empty array).
     */
    [[nodiscard]] size_t getArrayAddressRelative(uint16 arrayId, Ordinal oem) const {
        return arrayDefs.get(arrayId).getOEMAddress(oem);
    }

    void runOnOverflowCallback() { this->onOverflowCallback(); }

    bool usesP2pDma() const { return useP2pDma; }

    // On systems with unified memory (where GPU and CPU share the same memory, i.e. Nvidia Orin) P2P DMA is
    // functionally equivalent to regular DMA, only to memory regions which are owned by the GPU.
    // The only reason we don't return usesP2pDma==false is that because we shouldn't register this memory again.
    bool usesDmaBuf() const { return useP2pDma && !usesUnifiedMemory; }

private:
    // How many bytes will be allocated in a single kernel call.
    // The value was determined heuristically in a way so to avoid allocation time overhead and to make sure
    // that the watchdog thread is possibly not blocked by the address space lock too long.
    static constexpr size_t ALLOC_CHUNK_SIZE = 512 * 1024 * 1024; // 512 MiB

    void releaseDataBuffer() {
        if (dataBuffer == nullptr) {
            return;
        }
        if (useP2pDma) {
            auto &cuda = CudaRuntime::instance();
            if (usesUnifiedMemory) {
                cuda.freeHost(dataBuffer);
            } else {
                cuda.free(dataBuffer);
            }
        } else {
            freeChunked(dataBuffer, dataBufferSize);
        }
    }

    /**
     * Allocates a contiguous virtual memory region of the given totalSize,
     * committing it in smaller chunks to avoid holding the kernel's mmap_lock
     * (Linux) or address space lock (Windows) for too long.
     *
     * The intention of this function is to avoid any time-critical (e.g. watchdog) to be blocked during
     * the allocation stage, i.e. give the possibility to wake up between (chunk) allocations.
     *
     * NOTE: mmap (Linux) and VirtualAlloc (Windows) return page-aligned addresses
     * (>= 4096 bytes). Static assert ensures ALIGNMENT doesn't exceed this guarantee.
     */
    static void *mallocChunked(size_t totalSize, size_t chunkSize) {
        static_assert(ALIGNMENT <= 4096,
                      "ALIGNMENT exceeds page size; mmap/VirtualAlloc may not satisfy it");
#ifdef _WIN32
        // Reserve contiguous VIRTUAL address space.
        void *base = VirtualAlloc(nullptr, totalSize, MEM_RESERVE, PAGE_NOACCESS);
        if (!base) {
            throw std::bad_alloc();
        }
        // Allocate chunks.
        for (size_t offset = 0; offset < totalSize; offset += chunkSize) {
            size_t thisChunk = std::min(chunkSize, totalSize - offset);
            void *committed = VirtualAlloc(static_cast<char *>(base) + offset, thisChunk, MEM_COMMIT, PAGE_READWRITE);
            if (!committed) {
                VirtualFree(base, 0, MEM_RELEASE);
                throw std::bad_alloc();
            }
        }
        return base;
#else
        // Reserve contiguous VIRTUAL address space (PROT_NONE = no access, no physical pages).
        void *base = ::mmap(nullptr, totalSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (base == MAP_FAILED) {
            throw std::bad_alloc();
        }
        // Allocate chunks (== remapping with read/write permissions).
        for (size_t offset = 0; offset < totalSize; offset += chunkSize) {
            size_t thisChunk = std::min(chunkSize, totalSize - offset);
            void *committed = ::mmap(static_cast<char *>(base) + offset, thisChunk, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_POPULATE, -1, 0);
            if (committed == MAP_FAILED) {
                ::munmap(base, totalSize);
                throw std::bad_alloc();
            }
        }
        return base;
#endif
    }

    /**
     * Frees memory allocated with the mallocChunked.
     */
    static void freeChunked(void *base, size_t totalSize) {
        if (!base) {
            return;
        }
#ifdef _WIN32
        UNREFERENCED_PARAMETER(totalSize);
        VirtualFree(base, 0, MEM_RELEASE);
#else
        ::munmap(base, totalSize);
#endif
    }

    /**
     * Throws IllegalStateException when the buffer is in invalid state.
     *
     * @return true if the queue execution should continue, false otherwise.
     */
    void validateState() {
        if (this->state == State::INVALID) {
            throw ::arrus::IllegalStateException("The buffer is in invalid state "
                                                 "(probably some data transfer overflow happened).");
        } else if (this->state == State::SHUTDOWN) {
            throw ::arrus::IllegalStateException("The data buffer has been turned off.");
        }
    }

    /**
     * Creates the expected value of the pattern when all the data was properly transferred to this buffer.
     */
    static Accumulator createElementReadyPattern(const Tuple<Us4ROutputBufferArrayDef> &arrays, size_t noems) {
        // accumulator for each array
        std::vector<Accumulator> accumulators;
        for (auto &array : arrays.getValues()) {
            Accumulator accumulator((1ul << noems) - 1);
            for (size_t oem = 0; oem < noems; ++oem) {
                if (array.getOEMSize(ARRUS_SAFE_CAST(oem, Ordinal)) == 0) {
                    accumulator &= ~(1ul << oem);
                }
            }
            accumulators.push_back(accumulator);
        }
        // OEM is active when at least array is produced by this OEM.
        Accumulator result = 0;
        for (const auto &a : accumulators) {
            result = result | a;
        }
        return result;
    }

    /**
     * Returns the size of the whole element, i.e. the sum of the sizes of all arrays (the number of bytes).
     */
    static size_t calculateElementSize(const Tuple<Us4ROutputBufferArrayDef> &arrays) {
        size_t result = 0;
        for (auto &array : arrays.getValues()) {
            result += array.getSize();
        }
        return result;
    }

    void createElements(const Tuple<Us4ROutputBufferArrayDef> &arrayDefinitions, uint16 elementReadyPattern,
                        unsigned nElements, size_t elementSizeBytes) {
        for (unsigned i = 0; i < nElements; ++i) {
            std::vector<framework::NdArray> arraysVector;
            for (const Us4ROutputBufferArrayDef &arrayDef : arrayDefinitions.getValues()) {
                size_t elementOffset = i * elementSizeBytes;
                size_t arrayOffset = elementOffset + arrayDef.getAddress();
                auto arrayAddress = reinterpret_cast<DataType *>(reinterpret_cast<int8 *>(dataBuffer) + arrayOffset);
                auto def = arrayDef.getDefinition();
                // The data pointer lives in GPU memory when P2P DMA is used (and the
                // GPU does not use unified memory); otherwise it is plain host memory.
                DeviceType placementType = (useP2pDma && !usesUnifiedMemory)
                                               ? DeviceType::GPU
                                               : DeviceType::CPU;
                DeviceId deviceId(placementType, 0);
                framework::NdArray array{arrayAddress, def.getShape(), def.getDataType(), deviceId};
                arraysVector.emplace_back(std::move(array));
            }
            Tuple<framework::NdArray> arrays = Tuple<framework::NdArray>{arraysVector};
            elements.push_back(std::make_shared<Us4ROutputBufferElement>(i, arrays, elementReadyPattern));
        }
    }

    std::mutex mutex;
    /** A size of a single element IN number of BYTES. */
    size_t elementSize;
    /** Total size of dataBuffer in bytes. */
    size_t dataBufferSize{0};
    /**  Total size in the number of elements. */
    int16 *dataBuffer{nullptr};
    /** Host buffer elements */
    std::vector<Us4ROutputBufferElement::SharedHandle> elements;
    /** Array offsets, in bytes. The is an offset relative to the beginning of each element. */
    std::vector<size_t> arrayOffsets;
    /** OEM data offset, relative to the beginning of array, in bytes. */
    std::vector<size_t> arrayOEMOffsets;
    // Callback that should be called once new data arrive.
    framework::OnNewDataCallback onNewDataCallback;
    framework::OnOverflowCallback onOverflowCallback{[]() {}};
    framework::OnShutdownCallback onShutdownCallback{[]() {}};

    // State management
    enum class State { RUNNING, SHUTDOWN, INVALID };
    State state{State::RUNNING};
    bool stopOnOverflow{true};
    Tuple<Us4ROutputBufferArrayDef> arrayDefs;

    /** Whether to use P2P DMA. */
    bool useP2pDma;
    /** Whether the Nvidia GPU uses unified memory, i.e. the same memory space as the CPU. */
    bool usesUnifiedMemory{false};
};

class Us4ROutputBufferBuilder {
public:
    Us4ROutputBufferBuilder &setStopOnOverflow(bool value) {
        stopOnOverflow = value;
        return *this;
    }

    Us4ROutputBufferBuilder &setNumberOfElements(unsigned n) {
        nElements = n;
        return *this;
    }

    Us4ROutputBufferBuilder &setUseP2pDma(bool value) {
        useP2pDma = value;
        return *this;
    }

    Us4ROutputBufferBuilder &setLayoutTo(const std::vector<Us4OEMBuffer> &buffers) {
        if (buffers.empty() || buffers.at(0).getNumberOfArrays() == 0) {
            // No arrays are acquired here.
            return *this;
        }
        ArrayId nArrays = buffers.at(0).getNumberOfArrays();
        noems = ARRUS_SAFE_CAST(buffers.size(), unsigned);

        std::vector<Us4ROutputBufferArrayDef> result;
        // Array -> shape
        std::vector<framework::NdArrayDef::Shape> shapes;
        // Array -> OEM -> shape
        std::vector<std::vector<framework::NdArrayDef::Shape>> partShapes(nArrays);
        // Array -> OEM -> size
        std::vector<std::vector<size_t>> oemSizes(nArrays);
        for (auto &v : partShapes) {
            v.resize(noems);
        }
        for (auto &v : oemSizes) {
            v.resize(noems);
        }
        // Transpose
        for (Ordinal oem = 0; oem < (Ordinal) buffers.size(); ++oem) {
            const auto &buffer = buffers.at(oem);
            for (ArrayId arrayId = 0; arrayId < buffer.getNumberOfArrays(); ++arrayId) {
                const auto &oemArrayDef = buffer.getArrayDefs().at(arrayId);
                const auto ndarrayDef = oemArrayDef.getDefinition();
                ARRUS_REQUIRES_TRUE(ndarrayDef.getDataType() == Us4ROutputBuffer::ARRAY_DATA_TYPE,
                                    "Unexpected OEM array data type.");
                partShapes.at(arrayId).at(oem) = ndarrayDef.getShape();
                oemSizes.at(arrayId).at(oem) = oemArrayDef.getSize();
            }
        }
        // Concatenate shape of each array (concatenate array elements produced by each us4OEM)
        for (const auto &arrayShapes : partShapes) {
            shapes.emplace_back(concatenate(arrayShapes));
        }
        size_t address = 0;
        for (ArrayId arrayId = 0; arrayId < nArrays; ++arrayId) {
            framework::NdArrayDef definition{shapes.at(arrayId), Us4ROutputBuffer::ARRAY_DATA_TYPE};
            result.emplace_back(definition, address, oemSizes.at(arrayId));
            address += definition.getSize();
        }

        arrayDefs = Tuple<Us4ROutputBufferArrayDef>(result);
        return *this;
    }

    Us4ROutputBuffer::SharedHandle build() {
        return std::make_shared<Us4ROutputBuffer>(arrayDefs, nElements, stopOnOverflow, noems, useP2pDma);
    }

private:
    /**
     * Concatenates shapes. If shape is empty (empty array), skip.
     * @param parts: parts of a given array of a given OEM
     */
    framework::NdArrayDef::Shape concatenate(const std::vector<framework::NdArrayDef::Shape> &parts) {
        // Find first non-empty shape and use it as a starting point.
        auto start = std::find_if(std::begin(parts), std::end(parts), [](const auto &shape) { return !shape.empty(); });
        if (start == std::end(parts)) {
            // all parts empty, return empty shape
            return framework::NdArrayDef::Shape{};
        }
        size_t pos = std::distance(std::begin(parts), start);
        auto ref = start->getValues();
        size_t chAx = ref.size() - 1;
        constexpr size_t sampAx = 0;
        auto nCh = static_cast<unsigned>(ref[chAx]);
        for (size_t i = pos + 1; i < parts.size(); ++i) {
            const auto &part = parts.at(i);
            ARRUS_REQUIRES_TRUE_IAE(nCh == part.get(chAx),
                                    "Each us4OEM buffer element should have the same number of channels.");
            ref[sampAx] += static_cast<unsigned>(part.get(sampAx));
        }
        return framework::NdArray::Shape{ref};
    }
    Tuple<Us4ROutputBufferArrayDef> arrayDefs;
    unsigned noems{0};
    unsigned nElements{0};
    bool stopOnOverflow{false};
    bool useP2pDma{false};
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4ROUTPUTBUFFER_H
