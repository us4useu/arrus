#ifndef ARRUS_CORE_API_OPS_US4R_H
#define ARRUS_CORE_API_OPS_US4R_H

#include <utility>

#include "TxRxSequence.h"
#include "arrus/core/api/framework/FifoBufferSpec.h"

namespace arrus::ops::us4r {

enum class WorkMode {
    ASYNC,
    SYNC
};

class Scheme {
public:
    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const framework::FifoBufferSpec &outputBuffer)
        : txRxSequence(std::move(txRxSequence)), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer) {}

    const TxRxSequence &getTxRxSequence() const {
        return txRxSequence;
    }

    uint16 getRxBufferSize() const {
        return rxBufferSize;
    }

    const framework::FifoBufferSpec &getOutputBuffer() const {
        return outputBuffer;
    }

private:
    TxRxSequence txRxSequence;
    uint16 rxBufferSize;
    ::arrus::framework::FifoBufferSpec outputBuffer;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_H
