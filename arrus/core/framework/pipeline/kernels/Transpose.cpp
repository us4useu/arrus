#include "Transpose.h"

namespace arrus_example_imaging {

TransposeKernel::TransposeKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    size_t rank = inputShape.size();
    if(rank < 2) {
        throw std::runtime_error("The transposition can be performed on arrays with at least two arrays.");
    }
    nColumns = inputShape[rank-1];
    nRows = inputShape[rank-2];
    nMatrices = 1;
    if(rank > 2) {
        for(int i = 0; i < rank-2; ++i) {
            nMatrices *= inputShape[i];
        }
    }
    DataShape outputShape;
    outputShape = inputShape;
    std::swap(outputShape[rank-1], outputShape[rank-2]);
    ctx.setOutput(NdArrayDef{outputShape, inputType});
}

void TransposeKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), nMatrices, nRows, nColumns, ctx.getStream());
}

REGISTER_KERNEL_OP(OPERATION_CLASS_ID(Transpose), TransposeKernel)

}
