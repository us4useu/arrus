#ifndef ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H
#define ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H

#include <memory>
#include <functional>
#include "NdArray.h"

namespace arrus::framework {

class DataBufferElement {
public:
    using SharedHandle = std::shared_ptr<DataBufferElement>;

    virtual ~DataBufferElement() = default;

    virtual void release() = 0;

    virtual NdArray& getData() = 0;

    /**
     * @return size of the element in bytes
     */
    virtual size_t getSize() = 0;

    virtual size_t getPosition() = 0;
};

/**
 * FIFO (first in, first out) buffer.
 */
class DataBuffer {
public:
    using Handle = std::unique_ptr<DataBuffer>;
    using SharedHandle = std::shared_ptr<DataBuffer>;

    virtual ~DataBuffer() = default;

    /**
     * Returns number of elements the buffer contains.
     */
    virtual size_t getNumberOfElements() const = 0;

    /**
     * Returns a pointer to selected element buffer.
     *
     * @param i number of buffer element
     * @return a pointer to the buffer element
     */
    virtual std::shared_ptr<DataBufferElement> getElement(size_t i) = 0;

    /**
     * Returns size of a single buffer element, that is the number of values of a given data type.
     */
    virtual size_t getElementSize() const = 0;


};

}

#endif //ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H
