#ifndef CPP_EXAMPLE_IMAGING_KERNELS_ENVELOPEDETECTION_H
#define CPP_EXAMPLE_IMAGING_KERNELS_ENVELOPEDETECTION_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/EnvelopeDetection.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class EnvelopeDetectionFunctor {
public:
    void operator()(NdArray &output, const NdArray &input, cudaStream_t stream);
};

class EnvelopeDetectionKernel : public Kernel {
public:
    explicit EnvelopeDetectionKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    EnvelopeDetectionFunctor impl;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_ENVELOPEDETECTION_H
