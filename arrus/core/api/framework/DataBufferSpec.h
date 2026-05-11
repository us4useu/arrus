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

    DataBufferSpec()
        : bufferType(Type::FIFO), nElements(2),
          placement(::arrus::devices::DeviceId(::arrus::devices::DeviceType::CPU, 0)) {}

    /**
     * Data buffer specification constructor.
     *
     * @param bufferType buffer type
     * @param nElements number of elements (a single element of the buffer is an output of a single tx/rx sequence execution)
     */
    DataBufferSpec(Type bufferType, const unsigned &nElements)
        : bufferType(bufferType), nElements(nElements),
          placement(::arrus::devices::DeviceId(::arrus::devices::DeviceType::CPU, 0)) {}

    /**
     * Data buffer specification constructor.
     *
     * @param bufferType buffer type
     * @param nElements number of elements
     * @param placement device on which the buffer should be allocated. Allowed values: (CPU, 0), (GPU, 0). Default: (CPU, 0).
     */
    DataBufferSpec(Type bufferType, const unsigned &nElements, const ::arrus::devices::DeviceId &placement)
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

    /**
     * Returns the device on which the buffer is placed.
     */
    const ::arrus::devices::DeviceId &getPlacement() const {
        return placement;
    }

private:
    Type bufferType;
    unsigned nElements{};
    ::arrus::devices::DeviceId placement;
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_DATABUFFERSPEC_H
