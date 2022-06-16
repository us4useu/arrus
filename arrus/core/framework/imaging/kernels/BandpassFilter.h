#ifndef CPP_EXAMPLE_IMAGING_KERNELS_BANDPASSFILTER_H
#define CPP_EXAMPLE_IMAGING_KERNELS_BANDPASSFILTER_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/BandpassFilter.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class BandpassFilterFunctor {
public:
    BandpassFilterFunctor() = default;

    BandpassFilterFunctor(const NdArray &coeffs);

    void operator()(NdArray &output, const NdArray &input,
                    unsigned totalNSamples, unsigned nCoefficients, unsigned nSamples,
                    cudaStream_t stream);
};

class BandpassFilterKernel : public Kernel {
public:
    explicit BandpassFilterKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    BandpassFilterFunctor impl;
    unsigned totalNSamples, nCoefficients, nSamples;
    NdArray coefficients;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_BANDPASSFILTER_H
