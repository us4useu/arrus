#ifndef CPP_EXAMPLE_IMAGING_KERNELS_TRANSPOSE_H
#define CPP_EXAMPLE_IMAGING_KERNELS_TRANSPOSE_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Transpose.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class TransposeFunctor {
public:
    void operator()(NdArray &output, const NdArray &input,
                    unsigned nMatrices, unsigned nRows, unsigned nColumns,
                    cudaStream_t stream);
};

class TransposeKernel : public Kernel {
public:
    explicit TransposeKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    TransposeFunctor impl;
    unsigned nMatrices, nRows, nColumns;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_TRANSPOSE_H
