#ifndef ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H
#define ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

class HostBuffer {
public:
    using Handle = std::unique_ptr<HostBuffer>;
    using SharedHandle = std::shared_ptr<HostBuffer>;
    static constexpr long long INF_TIMEOUT = -1;

    virtual ~HostBuffer() = default;

    /**
     * @param timeout -1 means infinity timeout
     * @return
     */
    virtual int16* tail(long long timeout) = 0;

    virtual int16* head(long long timeout) = 0;

    virtual size_t tailAddress(long long timeout) {
        return (size_t)tail(timeout);
    }

    virtual size_t headAddress(long long timeout) {
        return (size_t)head(timeout);
    }

    virtual void releaseTail(int firing) = 0;

    virtual unsigned short getNumberOfElements() const = 0;

    virtual size_t getElementSize() const = 0;

    virtual int16* getElement(size_t i) = 0;

    virtual size_t getElementAddress(size_t i) {
        return (size_t)getElement(i);
    }
};

}

#endif //ARRUS_CORE_API_OPS_US4R_HOSTBUFFER_H
