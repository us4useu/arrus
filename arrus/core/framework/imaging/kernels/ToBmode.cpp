#include "ToBmode.h"

namespace arrus_example_imaging {

ToBModeKernel::ToBModeKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    // TODO make sure the below parameters are scalars
    minDbLimit = ctx.getParamArray("minDbLimit").get<float>(0);
    maxDbLimit = ctx.getParamArray("maxDbLimit").get<float>(0);
    ctx.setOutput(NdArrayDef{inputShape, DataType::UINT8});
}
void ToBModeKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), minDbLimit, maxDbLimit, ctx.getStream());
}

REGISTER_KERNEL_OP(OPERATION_CLASS_ID(ToBMode), ToBModeKernel)

}