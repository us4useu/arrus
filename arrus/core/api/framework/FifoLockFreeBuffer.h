#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOLOCKFREEBUFFER_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOLOCKFREEBUFFER_H

#include "DataBuffer.h"

namespace arrus::framework {

/**
 * A callback to be called once new data arrives.
 */
using OnNewDataCallback = std::function<void(const DataBufferElement::SharedHandle& )>;
using OnOverflowCallback = std::function<void()>;
using OnShutdownCallback = std::function<void()>;

/**
 * First in, first out buffer. The buffer is lock-free from the producer's point of view --
 * when data overflow happens (no more space in the buffer for new data), the device is stopped
 * and overflow callback is called.
 */
class FifoLockFreeBuffer: public DataBuffer {
public:
    using Handle = std::unique_ptr<FifoLockFreeBuffer>;
    using SharedHandle = std::shared_ptr<FifoLockFreeBuffer>;

    /**
     * Registers callback, that should be called once new data arrives at the buffer head.
     * The callback has an access to the latest data.
     *
     * Free the provided buffer element using `release` function when the data is no longer needed.
     *
     * The callback is required.
     */
    virtual void registerOnNewDataCallback(OnNewDataCallback &callback) = 0;

    /**
     * Registers callback, that will be called once buffer overflow happens.
     *
     * The callback function is optional, by default nop is performed.
     */
    virtual void registerOnOverflowCallback(OnOverflowCallback &callback) = 0;

    /**
     * Registers callback, that will be called once buffer is shutdown.
     *
     * Buffer shutdown is preformed when the device is stopped.
     * The callback function is optional, by default no ope is set.
     */
    virtual void registerShutdownCallback(OnShutdownCallback &callback) = 0;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOLOCKFREEBUFFER_H
