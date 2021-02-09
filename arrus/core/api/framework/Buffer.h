#ifndef ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H
#define ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H

#include <memory>
#include <functional>
#include "NdArray.h"

namespace arrus::framework {

/**
 * A buffer element.
 */
class BufferElement {
public:
    using SharedHandle = std::shared_ptr<BufferElement>;

    virtual ~BufferElement() = default;

    /**
     * Releases given buffer element for further data acquisitions.
     */
    virtual void release() = 0;

    /**
     * Returns output data array.
     */
    virtual NdArray& getData() = 0;

    /**
     * @return size of the element in bytes
     */
    virtual size_t getSize() = 0;

    /**
     * Returns position of the element in the data buffer.
     */
    virtual size_t getPosition() = 0;
};

/**
 * A buffer.
 */
class Buffer {
public:
    using Handle = std::unique_ptr<Buffer>;
    using SharedHandle = std::shared_ptr<Buffer>;

    virtual ~Buffer() = default;

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
    virtual std::shared_ptr<BufferElement> getElement(size_t i) = 0;

    /**
     * Returns size of a single buffer element, that is the number of values of a given data type.
     */
    virtual size_t getElementSize() const = 0;


};

}

#endif //ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H
