#ifndef ARRUS_ARRUS_CORE_KERNELS_KERNEL_H
#define ARRUS_ARRUS_CORE_KERNELS_KERNEL_H

#include <memory>

namespace arrus::kernels {

class Kernel {
public:
    using SharedHandle = std::shared_ptr<Kernel>;

    virtual void process() = 0;
};

}

#endif //ARRUS_ARRUS_CORE_KERNELS_KERNEL_H
