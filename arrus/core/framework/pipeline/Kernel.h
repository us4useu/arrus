#ifndef CPP_EXAMPLE_KERNELS_KERNEL_CUH
#define CPP_EXAMPLE_KERNELS_KERNEL_CUH

#include "KernelConstructionContext.h"
#include "KernelExecutionContext.h"

#include <memory>
namespace arrus_example_imaging {
class Kernel {
public:
    typedef std::unique_ptr<Kernel> Handle;

    explicit Kernel(KernelConstructionContext &ctx) {}

    virtual void process(KernelExecutionContext &ctx) = 0;
};
}// namespace arrus_example_imaging
#endif//CPP_EXAMPLE_KERNELS_KERNEL_CUH
