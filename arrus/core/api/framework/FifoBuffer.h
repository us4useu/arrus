#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFER_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFER_H

namespace arrus::framework {

class FifoBuffer {
    virtual ~FifoBuffer() = default;


    enum class Mode {
        SYNC,
        /** Async mode - lock-free from the producer's point of view. */
        ASYNC
    };
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFER_H
