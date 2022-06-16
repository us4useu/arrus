#include "BandpassFilter.h"

namespace arrus_example_imaging {

BandpassFilterKernel::BandpassFilterKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    size_t rank = inputShape.size();

    totalNSamples = 1;
    for(auto size : inputShape) {
        totalNSamples *= size;
    }
    nSamples = inputShape[rank-1];

    coefficients = ctx.getParamArray("coefficients");
    if(coefficients.getDataType() != DataType::FLOAT32) {
        throw std::runtime_error("The filter coefficients must be float32. ");
    }
    if(coefficients.isGpu()) {
        throw std::runtime_error("Filter coefficients must be located in host memory.");
    }
    if(coefficients.getShape().size() > 1) {
        throw std::runtime_error("Filter coefficients must be a vector of values (detected rank: "
                                 + std::to_string(coefficients.getShape().size())
                                 + ").");
    }
    impl = BandpassFilterFunctor{coefficients};
    nCoefficients = coefficients.getNumberOfElements();
    ctx.setOutput(NdArrayDef{inputShape, DataType::FLOAT32});
}

void BandpassFilterKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(),
         totalNSamples, nCoefficients, nSamples,
         ctx.getStream());
}

REGISTER_KERNEL_OP(OPERATION_CLASS_ID(BandpassFilter), BandpassFilterKernel)

}
