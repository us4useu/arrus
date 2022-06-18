#ifndef CPP_EXAMPLE_IMAGING_KERNELS_RECONSTRUCTHRI_H
#define CPP_EXAMPLE_IMAGING_KERNELS_RECONSTRUCTHRI_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/BandpassFilter.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class ReconstructHriFunctor {
public:
    ReconstructHriFunctor() = default;

    ReconstructHriFunctor(const NdArray &zElemPos, const NdArray &xElemPos, const NdArray &elementTang);

    void operator()(NdArray &output, const NdArray &input,
                    const NdArray &zPix, const NdArray &xPix,
                    const NdArray &txFocuses, const NdArray &txAngles,
                    const NdArray &txApertureCenterZ, const NdArray &txApertureCenterX,
                    const NdArray &txApertureFirstElement, const NdArray &txApertureLastElement,
                    const NdArray &rxApertureOrigin,
                    unsigned nElements, float sos, float fs, float fn,
                    float minRxTang, float maxRxTang,
                    float initDelay,
                    cudaStream_t stream);
};

class ReconstructHriKernel : public Kernel {
public:
    explicit ReconstructHriKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    ReconstructHriFunctor impl;
    NdArray zElemPos, xElemPos, elementTang;
    NdArray zPix, xPix;
    NdArray txFocuses, txAngles;
    NdArray txApertureCenterZ, txApertureCenterX;
    NdArray txApertureFirstElement, txApertureLastElement;
    NdArray rxApertureOrigin;
    unsigned nElements;
    float sos, fs, fn, minRxTang, maxRxTang, initDelay;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_RECONSTRUCTHRI_H
