#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOLOCKFREEBUFFER_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOLOCKFREEBUFFER_H

#include "DataBuffer.h"

namespace arrus::framework {

/**
 * A callback to be called once new data arrives.
 */
using OnNewDataCallback = std::function<void(const DataBufferElement::SharedHandle& )>;
using OnOverflowCallback = std::function<void()>;

/**
 * First in, first out buffer.
 */
class FifoLockfreeBuffer: public DataBuffer {
public:
    /**
     * Registers callback, that should be called once new data arrives at the buffer head.
     * The callback has an access to the latest data.
     *
     * Free the provided buffer element using `release` function when the data is no longer needed.
     */
    virtual void registerOnNewDataCallback(OnNewDataCallback &callback) = 0;

    /**
     * Registers callback, that will be called once buffer overflow happens.
     */
    virtual void registerOnOverflowCallback(OnOverflowCallback &callback) = 0;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOLOCKFREEBUFFER_H
