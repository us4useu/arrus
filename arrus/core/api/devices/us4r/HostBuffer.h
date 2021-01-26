#ifndef ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H
#define ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H

#include "arrus/core/api/common/types.h"

namespace arrus::devices {



class HostBuffer {
public:
    using Handle = std::unique_ptr<HostBuffer>;
    using SharedHandle = std::shared_ptr<HostBuffer>;


    virtual ~HostBuffer() = default;

    virtual unsigned short getNumberOfElements() const = 0;

    virtual size_t getElementSize() const = 0;
};

}

#endif //ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H
