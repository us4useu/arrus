#include "RemapToLogicalOrder.h"

#include <arrus/core/api/arrus.h>

namespace arrus_example_imaging {

RemapToLogicalOrderKernel::RemapToLogicalOrderKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto fcm = ctx.getInputMetadata()->getObject<arrus::devices::FrameChannelMapping>("frameChannelMapping");

    // Determining output dimensions.
    auto rawSequence = ctx.getInputMetadata()->getObject<arrus::ops::us4r::TxRxSequence>("rawSequence");
    // TODO Note: assumption that all TxRxs have the same number of samples. Validate that.
    auto [startSample, endSample] = rawSequence->getOps()[0].getRx().getSampleRange();
    nSequences = rawSequence->getNRepeats();
    nFrames = fcm->getNumberOfLogicalFrames();
    nSamples = endSample-startSample;
    nChannels = fcm->getNumberOfLogicalChannels();

    // Prepare auxiliary arrays.
    std::vector<uint16_t> frames(nFrames*nChannels);
    std::vector<int8_t> channels(nFrames*nChannels);
    std::vector<uint8_t> us4oems(nFrames*nChannels);

    for(size_t frame = 0; frame < fcm->getNumberOfLogicalFrames(); ++frame) {
        for(int channel = 0; channel < fcm->getNumberOfLogicalChannels(); ++channel) {
            auto addr = fcm->getLogical(frame, channel);
            auto idx = frame*nChannels+channel;
            frames[idx] = addr.getFrame();
            channels[idx] = addr.getChannel();
            us4oems[idx] = addr.getUs4oem();
        }
    }
    fcmFrames = NdArray::asarray(frames, true).reshape({nFrames, nChannels});
    fcmChannels = NdArray::asarray(channels, true).reshape({nFrames, nChannels});
    fcmUs4oems = NdArray::asarray(us4oems, true).reshape({nFrames, nChannels});
    frameOffsets = NdArray::asarray(fcm->getFrameOffsets());
    nFramesUs4OEM = NdArray::asarray(fcm->getNumberOfFrames());

    ctx.setOutput(NdArrayDef{{nSequences, nFrames, nSamples, nChannels}, ctx.getInput().getType()});
}

void RemapToLogicalOrderKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), fcmFrames, fcmChannels, fcmUs4oems, frameOffsets, nFramesUs4OEM,
         nSequences, nFrames, nSamples, nChannels, ctx.getStream());
}

REGISTER_KERNEL_OP(OPERATION_CLASS_ID(RemapToLogicalOrder), RemapToLogicalOrderKernel);

}// namespace arrus_example_imaging