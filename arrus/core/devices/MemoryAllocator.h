#ifndef ARRUS_CORE_DEVICES_MEMORYALLOCATOR_H
#define ARRUS_CORE_DEVICES_MEMORYALLOCATOR_H

#include "arrus/core/devices/

namespace arrus::devices {

class MemoryAllocator {
    virtual ~MemoryAllocator() = default;
    virtual NdArray allocate(size_t size) = 0;
};

}

#endif//ARRUS_CORE_DEVICES_MEMORYALLOCATOR_H
