#ifndef ARRUS_ARRUS_CORE_SESSION_KERNELREGISTRY_H
#define ARRUS_ARRUS_CORE_SESSION_KERNELREGISTRY_H

#include "arrus/core/kernels/Kernel.h"
#include "arrus/core/kernels/KernelBuilder.h"
#include <unordered_map>

namespace arrus::kernels {

class KernelRegistry {



private:
    std::unordered_map<unsigned, ::arrus::kernels::KernelBuilder> kernels;
};

}


#endif //ARRUS_ARRUS_CORE_SESSION_KERNELREGISTRY_H
