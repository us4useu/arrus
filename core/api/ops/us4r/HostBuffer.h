#ifndef ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H
#define ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H

#include "arrus/core/api/common/types.h"

namespace arrus::ops::us4r {

class HostBuffer {
public:
    virtual ~HostBuffer() = default;
    virtual int16* tail() = 0;
    virtual void releaseTail() = 0;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H
