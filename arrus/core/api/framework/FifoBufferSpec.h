#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFERSPEC_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFERSPEC_H

namespace arrus::framework {

/**
 * FifoBuffer work mode.
 *
 * Determines behaviour of the buffer on a case of buffer overflow.
 */
enum class BufferWorkMode {
    /** Stop the device, mark the buffer as invalid (potential data overwrite happened). */
    ASYNC,
    /** Producer will wait until the buffer tail will be released. */
    SYNC
    // TODO CINELOOP
};

/**
 * Class describing output data buffer properties.
 */
class FifoBufferSpec {
public:

    /**
     * Creates buffer specification.
     *
     * @param workMode buffer work mode
     * @param nElements number of elements
     */
    FifoBufferSpec(BufferWorkMode workMode, const unsigned &nElements) : workMode(workMode), nElements(nElements) {}

    BufferWorkMode getWorkMode() const {
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
    BufferWorkMode workMode;
    unsigned nElements;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFERSPEC_H
