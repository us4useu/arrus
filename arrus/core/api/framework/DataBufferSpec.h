#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H

#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::framework {

/**
 * Class describing output data buffer properties.
 */
class DataBufferSpec {
public:

    /**
     * Buffer type.
     */
    enum class Type {
        /** First in first out buffer.*/
        FIFO
//    TODO CINELOOP
    };

    /**
     * Data buffer specification constructor.
     *
     * By default, the buffer will be allocated on CPU.
     *
     * @param bufferType buffer type
     * @param nElements number of elements (a single element of the buffer is an output of a single tx/rx sequence execution)
     */
    DataBufferSpec(Type bufferType, const unsigned &nElements)
        : DataBufferSpec(bufferType, nElements, ::arrus::devices::DeviceId(::arrus::devices::DeviceType::CPU, 0)) {}

    /**
     * Data buffer specification constructor.
     *
     * @param bufferType buffer type
     * @param nElements number of elements (a single element of the buffer is an output of a single tx/rx
     *   sequence execution)
     * @param placement on which device the buffer should be located
     */
    DataBufferSpec(Type bufferType, const unsigned &nElements, ::arrus::devices::DeviceId placement)
        : bufferType(bufferType), nElements(nElements), placement(placement) {}

    Type getType() const {
        return bufferType;
    }

    /**
     * Returns number of elements the buffer consists of.
     */
    unsigned getNumberOfElements() const {
        return nElements;
    }

    const devices::DeviceId &getPlacement() const {
        return placement;
    }

private:
    Type bufferType;
    unsigned nElements;
    ::arrus::devices::DeviceId placement;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
