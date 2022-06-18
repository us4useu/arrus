#ifndef CPP_EXAMPLE_IMAGING_KERNELS_DIGITALDOWNCONVERSION_H
#define CPP_EXAMPLE_IMAGING_KERNELS_DIGITALDOWNCONVERSION_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/DigitalDownConversion.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class QuadratureDemodulationFunctor {
public:
    QuadratureDemodulationFunctor() = default;

    /**
     * @param nSamples the number of samples in a single line (last dimension)
     * @param totalNSamples total number of elements in NdArray
     */
    void operator()(NdArray &output, const NdArray &input,
                    unsigned nSamples,
                    unsigned totalNSamples, float sampleCoeff,
                    cudaStream_t stream);
};

class LowPassFilterFunctor {
public:
    LowPassFilterFunctor() = default;

    LowPassFilterFunctor(const NdArray &coeffs);

    void operator()(NdArray &output, const NdArray &input,
                    unsigned nSamples, unsigned totalNSamples, unsigned nCoefficients,
                    cudaStream_t stream);
};

class DecimationFunctor {
public:
    DecimationFunctor() = default;

    void operator()(NdArray &output, const NdArray &input,
                    const unsigned totalOutputNSamples, const unsigned nSamples,
                    const unsigned decimationFactor,
                    cudaStream_t stream);
};



class DigitalDownConversionKernel : public Kernel {
public:
    explicit DigitalDownConversionKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    QuadratureDemodulationFunctor demodImpl;
    LowPassFilterFunctor lpImpl;
    DecimationFunctor decImpl;
    unsigned nSamples, totalNSamples;
    // Demodulation
    float demodSampleCoeff;
    NdArray demodulatedData;
    // Low-pass filter
    NdArray lpCoeffs;
    unsigned lpnCoefficients;
    NdArray filteredData;
    // Decimation
    unsigned decimationFactor;
    unsigned totalOutputNSamples;
};
}


#endif//CPP_EXAMPLE_IMAGING_KERNELS_DIGITALDOWNCONVERSION_H
