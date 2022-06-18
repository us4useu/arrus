#ifndef CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER_H
#define CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/RemapToLogicalOrder.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class RemapToLogicalOrderFunctor {
public:
    static constexpr int BLOCK_TILE_DIM = 32;
    void operator()(NdArray& output, const NdArray& input,
                    const NdArray& fcmFrames, const NdArray& fcmChannels, const NdArray &fcmUs4oems,
                    const NdArray& frameOffsets, const NdArray& nFramesUs4OEM,
                    unsigned nSequences, unsigned nFrames, unsigned nSamples, unsigned nChannels,
                    cudaStream_t stream);
};

class RemapToLogicalOrderKernel : public Kernel {
public:
    explicit RemapToLogicalOrderKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    NdArray fcmFrames;
    NdArray fcmChannels;
    NdArray fcmUs4oems;
    NdArray frameOffsets;
    NdArray nFramesUs4OEM;
    // number of logical sequences, frames, ...
    unsigned nSequences, nFrames, nSamples, nChannels;
    RemapToLogicalOrderFunctor impl;
};



}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER_H
