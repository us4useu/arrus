#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFER_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFER_H

#include "Buffer.h"

namespace arrus::framework {

/**
 * A callback to be called once new data arrives.
 */
using OnNewDataCallback = std::function<void(const BufferElement::SharedHandle& )>;
/**
 * A callback to be called when data overflow happens.
 */
using OnOverflowCallback = std::function<void()>;
/**
 * A callback to be called when the buffer is shut down (e.g. when the session is stopped).
 */
using OnShutdownCallback = std::function<void()>;

/**
 * A data buffer. This interface allows to register callback function to be called when new data arrives.
 */
class DataBuffer: public Buffer {
public:
    using Handle = std::unique_ptr<DataBuffer>;
    using SharedHandle = std::shared_ptr<DataBuffer>;

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
     * The callback function is optional, by default nop is set.
     */
    virtual void registerShutdownCallback(OnShutdownCallback &callback) = 0;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFER_H
