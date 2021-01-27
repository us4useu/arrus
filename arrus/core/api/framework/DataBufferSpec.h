#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H

namespace arrus::framework {

/**
 * DataBuffer work mode.
 *
 * Determines behaviour of the buffer on a case of buffer overflow.
 */
enum class DataBufferType {
    /** Data buffer is lock-free from the producer's point of view, the buffer is marked as invalid when overflow happens. */
    FIFO_LOCKFREE,
//    /** Producer is blocked until the buffer tail will be released. */
//    TODO SYNC
//    TODO CINELOOP
};

/**
 * Class describing output data buffer properties.
 */
class DataBufferSpec {
public:

    /**
     * Creates buffer specification.
     *
     * @param bufferType buffer work mode
     * @param nElements number of elements
     */
    DataBufferSpec(DataBufferType bufferType, const unsigned &nElements) : workMode(bufferType), nElements(nElements) {}

    DataBufferType getWorkMode() const {
        return workMode;
    }

    /**
     * Returns number of elements the buffer consists of.
     * @return
     */
    unsigned getNumberOfElements() const {
        return nElements;
    }

private:
    DataBufferType workMode;
    unsigned nElements;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
