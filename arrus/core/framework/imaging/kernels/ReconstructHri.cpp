#include "ReconstructHri.h"

#include "imaging/pwi.h"

namespace arrus_example_imaging {

ReconstructHriKernel::ReconstructHriKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    // Probe parameters.
    auto probeModels = ctx.getInputMetadata()->getObject<std::vector<ProbeModelExt>>("probeModels");
    if(probeModels->size() > 1) {
        throw std::runtime_error("The reconstruct HRI kernel works only with a single linear array probe."
                                 "Please use ReconstructHriRca for RCA probes.");
    }
    auto &probe = probeModels->at(0);
    nElements = probe.getNumberOfElements();
    zElemPos = probe.getElementPositionZ();
    xElemPos = probe.getElementPositionLateral();
    elementTang = probe.getElementAngle().tang();

    impl = ReconstructHriFunctor(zElemPos, xElemPos, elementTang);
    // Output grid
    auto zPixCpu = ctx.getParamArray("zGrid");
    auto xPixCpu = ctx.getParamArray("xGrid");
    zPix = zPixCpu.toGpu();
    xPix = xPixCpu.toGpu();

    // Sequence parameters
    // TODO note: we are supporting PWI image reconstruction only here for now.
    auto sequence = ctx.getInputMetadata()->getObject<PwiSequence>("sequence");
    auto rawSequence = ctx.getInputMetadata()->getObject<arrus::ops::us4r::TxRxSequence>("rawSequence");
    auto nTx = sequence->getAngles().size();
    txAngles = NdArray::asarray(sequence->getAngles(), true);
    sos = sequence->getSpeedOfSound();
    fn = sequence->getPulse().getCenterFrequency();
    fs = ctx.getInputMetadata()->getValue("samplingFrequency");

    // TODO note: here are assumption to all other tx/rx parameters
    // tx focuses: all infs (only plane waves)
    txFocuses = NdArray::vector<float>(nTx, std::numeric_limits<float>::infinity()).toGpu();
    txApertureCenterX = NdArray::zeros<float>(nTx).toGpu();
    // TODO note: the below solution is not ideal in the case of curved probes with even number of elements.
    auto probeCenterZ = probe.getElementPositionZ().get<float>(nElements/2-1);
    txApertureCenterZ = NdArray::vector<float>(nTx, probeCenterZ).toGpu();
    txApertureFirstElement = NdArray::zeros<unsigned>(nTx).toGpu();
    txApertureLastElement = NdArray::vector<unsigned>(nTx, nElements-1).toGpu();
    rxApertureOrigin = NdArray::zeros<float>(nTx);

    // TODO parametrize the below
    minRxTang = -0.5f;
    maxRxTang = 0.5f;

    auto [startSample, endSample] = sequence->getSampleRange();

    // TODO Note: assuming that each Tx/Rx has the same center delay.
    unsigned centerChannel = (probe.getStartChannel()+probe.getStopChannel())/2;
    float delayA = rawSequence->getOps()[0].getTx().getDelays()[centerChannel-1];
    float delayB = rawSequence->getOps()[0].getTx().getDelays()[centerChannel];
    float centerDelay = (delayA + delayB)/2;

    float burstFactor = sequence->getPulse().getNPeriods() / (2 * fn);

    // TODO note: us4R specific
    initDelay = -(startSample/65e6f) // start sample (via nominal sampling frequency)
              + centerDelay
              + burstFactor;
    unsigned nSequences = ctx.getInput().getShape()[0];
    ctx.setOutput(NdArrayDef{{nSequences, (unsigned)xPix.getNumberOfElements(), (unsigned)zPix.getNumberOfElements()},
                             DataType::COMPLEX64});
}

void ReconstructHriKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), zPix, xPix, txFocuses, txAngles, txApertureCenterZ, txApertureCenterX,
         txApertureFirstElement, txApertureLastElement, rxApertureOrigin, nElements, sos, fs, fn,
         minRxTang, maxRxTang, initDelay, ctx.getStream());
}

REGISTER_KERNEL_OP(OPERATION_CLASS_ID(ReconstructHri), ReconstructHriKernel)

}