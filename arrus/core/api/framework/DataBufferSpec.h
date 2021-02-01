#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H

namespace arrus::framework {

/**
 * Class describing output data buffer properties.
 */
class DataBufferSpec {
public:

    /**
     * DataBuffer type.
     */
    enum class Type {
        /** First in first out buffer.*/
        FIFO
//    TODO CINELOOP
    };

    /**
     * Creates buffer specification.
     *
     * @param bufferType buffer type
     * @param nElements number of elements
     */
    DataBufferSpec(Type bufferType, const unsigned &nElements)
        : bufferType(bufferType), nElements(nElements) {}

    Type getType() const {
        return bufferType;
    }

    /**
     * Returns number of elements the buffer consists of.
     */
    unsigned getNumberOfElements() const {
        return nElements;
    }

private:
    Type bufferType;
    unsigned nElements;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
