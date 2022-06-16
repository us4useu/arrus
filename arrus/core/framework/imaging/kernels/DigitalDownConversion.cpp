#include "DigitalDownConversion.h"
#include <arrus/core/api/arrus.h>

#define _USE_MATH_DEFINES // TODO MSVC specific
#include <math.h>

namespace arrus_example_imaging {

DigitalDownConversionKernel::DigitalDownConversionKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    size_t rank = inputShape.size();
    totalNSamples = 1;
    for(auto value: inputShape) {
        totalNSamples *= value;
    }
    nSamples = inputShape[rank-1];

    // Qudarature demodulation.
    auto samplingFrequency = ctx.getInputMetadata()->getValue("samplingFrequency");
    // TODO note: assuming that each transmission uses the same tx frequency
    auto rawSequence = ctx.getInputMetadata()->getObject<arrus::ops::us4r::TxRxSequence>("rawSequence");
    auto txFrequency = rawSequence->getOps()[0].getTx().getExcitation().getCenterFrequency();
    demodSampleCoeff = -2.0f * M_PI * txFrequency / samplingFrequency;

    demodulatedData = NdArray{NdArrayDef{inputShape, DataType::COMPLEX64}, true};

    // Low-pass filter.
    // TODO note: duplicates BandPassFilter
    lpCoeffs = ctx.getParamArray("coefficients");
    if(lpCoeffs.getDataType() != DataType::FLOAT32) {
        throw std::runtime_error("The filter coefficients must be float32. ");
    }
    if(lpCoeffs.isGpu()) {
        throw std::runtime_error("Filter coefficients must be located in host memory.");
    }
    if(lpCoeffs.getShape().size() > 1) {
        throw std::runtime_error("Filter coefficients must be a vector of values (detected rank: "
                                 + std::to_string(lpCoeffs.getShape().size())
                                 + ").");
    }
    lpnCoefficients = lpCoeffs.getNumberOfElements();
    lpImpl = LowPassFilterFunctor(lpCoeffs);

    filteredData = NdArray{NdArrayDef{inputShape, DataType::COMPLEX64}, true};
    // Decimation
    decimationFactor = ctx.getParamArray("decimationFactor").get<unsigned>(0);
    auto outputNSamples = (unsigned) ceilf((float) nSamples / decimationFactor);
    DataShape outputShape = inputShape;
    outputShape[rank-1] = outputNSamples;
    totalOutputNSamples = 1;
    for(auto v: outputShape) {
        totalOutputNSamples *= v;
    }
    ctx.setOutput(NdArrayDef{outputShape, DataType::COMPLEX64});
    ctx.getOutputMetadataBuilder().setValue("samplingFrequency", samplingFrequency/decimationFactor);
}
void DigitalDownConversionKernel::process(KernelExecutionContext &ctx) {
    demodImpl(demodulatedData, ctx.getInput(), nSamples, totalNSamples, demodSampleCoeff, ctx.getStream());
    lpImpl(filteredData, demodulatedData, nSamples, totalNSamples, lpnCoefficients, ctx.getStream());
    decImpl(ctx.getOutput(), filteredData, totalOutputNSamples, nSamples, decimationFactor, ctx.getStream());
}
REGISTER_KERNEL_OP(OPERATION_CLASS_ID(DigitalDownConversion), DigitalDownConversionKernel)
}