#ifndef ARRUS_CORE_API_OPS_US4R_H
#define ARRUS_CORE_API_OPS_US4R_H

#include <utility>

#include "TxRxSequence.h"
#include "arrus/core/api/framework/DataBufferSpec.h"

namespace arrus::ops::us4r {

class Scheme {
public:
    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const framework::DataBufferSpec &outputBuffer)
        : txRxSequence(std::move(txRxSequence)), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer) {}

    const TxRxSequence &getTxRxSequence() const {
        return txRxSequence;
    }

    uint16 getRxBufferSize() const {
        return rxBufferSize;
    }

    const framework::DataBufferSpec &getOutputBuffer() const {
        return outputBuffer;
    }

private:
    TxRxSequence txRxSequence;
    uint16 rxBufferSize;
    ::arrus::framework::DataBufferSpec outputBuffer;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_H
