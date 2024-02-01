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
    enum class State {
        FREE, READY, INVALID
    };
    using SharedHandle = std::shared_ptr<BufferElement>;

    virtual ~BufferElement() = default;

    virtual void release() = 0;

    /**
     * Returns output data, with the given id (ordinal).
     *
     * In some cases (e.g. when running a sequence of TX/RX sequences)
     * the system/process can produce a tuple of arrays. This method
     * is kept for backward compatibility, and always gives the access
     * to the first element of the tuple.
     *
     * @return NdArray with data
     */
    virtual NdArray& getData(ArrayId id) = 0;

    /**
     * Returns output data, with the ordinal 0.
     *
     * See also getData(ArrayId ordinal).
     *
     * @return NdArray with data
     */
    virtual NdArray& getData() = 0;

    /**
     * Returns the size of this buffer element.
     *
     * NOTE: the size is equal to the sum of the sizes of all subelements
     * (e.g. in case of an element that stores a tuple of NdArrays, this
     * method will return the sum of all NdArrays in that tuple).
     *
     * @return size of the whole element in bytes
     */
    virtual size_t getSize() = 0;

    /**
     * Returns position of the element in the data buffer.
     */
    virtual size_t getPosition() = 0;

    virtual State getState() const = 0;
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

    virtual size_t getNumberOfElementsInState(BufferElement::State state) const = 0;

};

}

#endif //ARRUS_CORE_API_FRAMEWORK_FIFO_BUFFER_H
