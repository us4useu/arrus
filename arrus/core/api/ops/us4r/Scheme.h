#ifndef ARRUS_CORE_API_OPS_US4R_H
#define ARRUS_CORE_API_OPS_US4R_H

#include <utility>

#include "TxRxSequence.h"

namespace arrus::ops::us4r {

enum class WorkMode {
    ASYNC,
    SYNC
};

class Scheme {

public:
    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, uint16 hostBufferSize, WorkMode workMode)
            : txRxSequence(std::move(txRxSequence)),
              rxBufferSize(rxBufferSize),
              hostBufferSize(hostBufferSize),
              workMode(workMode) {}

    const TxRxSequence &getTxRxSequence() const {
        return txRxSequence;
    }

    uint16 getRxBufferSize() const {
        return rxBufferSize;
    }

    uint16 getHostBufferSize() const {
        return hostBufferSize;
    }

    WorkMode getWorkMode() const {
        return workMode;
    }

private:
    TxRxSequence txRxSequence;
    uint16 rxBufferSize;
    uint16 hostBufferSize;
    WorkMode workMode;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_H
