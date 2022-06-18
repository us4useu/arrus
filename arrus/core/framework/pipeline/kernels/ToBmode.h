#ifndef CPP_EXAMPLE_IMAGING_KERNELS_TOBMODE_H
#define CPP_EXAMPLE_IMAGING_KERNELS_TOBMODE_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/ToBMode.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class ToBModeFunctor {
public:
    void operator()(NdArray &output, const NdArray &input,
                    float minDbLimit, float maxDbLimit,
                    cudaStream_t stream);
};

class ToBModeKernel : public Kernel {
public:
    explicit ToBModeKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    ToBModeFunctor impl;
    float minDbLimit, maxDbLimit;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_TOBMODE_H
