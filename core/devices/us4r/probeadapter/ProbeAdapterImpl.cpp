#include "ProbeAdapterImpl.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "arrus/core/devices/us4r/common.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/common/aperture.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace arrus::devices {

using namespace ::arrus::ops::us4r;

ProbeAdapterImpl::ProbeAdapterImpl(DeviceId deviceId,
                                   ProbeAdapterModelId modelId,
                                   std::vector<Us4OEMImpl::RawHandle> us4oems,
                                   ChannelIdx numberOfChannels,
                                   ChannelMapping channelMapping)
    : ProbeAdapter(deviceId), logger(getLoggerFactory()->getLogger()),
      modelId(std::move(modelId)),
      us4oems(std::move(us4oems)),
      numberOfChannels(numberOfChannels),
      channelMapping(std::move(channelMapping)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

class ProbeAdapterTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    ProbeAdapterTxRxValidator(const std::string &componentName, ChannelIdx nChannels)
        : Validator(componentName), nChannels(nChannels) {}

    void validate(const TxRxParamsSequence &txRxs) override {
        // TODO each rx aperture in the us4oem should have the same size
        // TODO at least one operation in sequence is provided
        // so the output frame has a regular shape
        // the same applies to number of samples
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            auto firingStr = ::arrus::format("firing {}", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getRxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxDelays().size(), size_t(nChannels), firingStr);
        }
    }

private:
    ChannelIdx nChannels;
};

FrameChannelMapping::Handle
ProbeAdapterImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq, const TGCCurve &tgcSamples) {
    // Validate input sequence
    ProbeAdapterTxRxValidator validator(
        ::arrus::format("{} tx rx sequence", getDeviceId().toString()),
        numberOfChannels);
    validator.validate(seq);
    validator.throwOnErrors();

    // Split into multiple arrays.
    // us4oem, op number -> aperture/delays
    std::unordered_map<Ordinal, std::vector<BitMask>> txApertures, rxApertures;
    std::unordered_map<Ordinal, std::vector<std::vector<float>>> txDelaysList;

    // Here is an assumption, that each operation has the same size rx aperture.
    auto rxApertureSize = getNumberOfActiveChannels(seq[0].getRxAperture());
    auto nFrames = getNumberOfNoRxNOPs(seq);

    // (frame number, active rx channel number) -> module
    // active rx channel number here refers to the ordinal number of
    // active channel in the rx aperture; e.g. for rx aperture [0, 32, 42],
    // 0 has relative ordinal number 0, 32 has relative number 1,
    // 42 has relative number 2.
    Eigen::MatrixXi frameModule(nFrames, rxApertureSize);
    frameModule.setConstant(FrameChannelMapping::UNAVAILABLE);
    // (frame number, active rx channel number) -> active rx channel number for frame on the selected module
    Eigen::MatrixXi frameChannel(nFrames, rxApertureSize);
    frameChannel.setConstant(FrameChannelMapping::UNAVAILABLE);

    // Initialize helper arrays.
    for(size_t ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
        txApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        rxApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        txDelaysList.emplace(ordinal,
                             std::vector<std::vector<float>>(seq.size()));
    }

    // Split Tx, Rx apertures and tx delays into sub-apertures specific for
    // each us4oem module.
    uint32 opNumber = 0;

    uint32 frameNumber = 0;
    for(const auto &op : seq) {
        logger->log(LogSeverity::TRACE, arrus::format("Setting tx/rx {}", op));

        const auto &txAperture = op.getTxAperture();
        const auto &rxAperture = op.getRxAperture();
        const auto &txDelays = op.getTxDelays();
        ARRUS_REQUIRES_TRUE(txAperture.size() == rxAperture.size()
                            && txAperture.size() == numberOfChannels,
                            arrus::format(
                                "Tx and Rx apertures should have a size: {}",
                                numberOfChannels));

        for(size_t ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
            txApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
            rxApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
            txDelaysList[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
        }

        size_t activeAdapterCh = 0;
        bool isRxNop = true;
        std::vector<size_t> activeUs4oemCh(us4oems.size(), 0);
        for(size_t ach = 0; ach < numberOfChannels; ++ach) {
            const auto[dstModule, dstChannel] = channelMapping[ach];
            txApertures[dstModule][opNumber][dstChannel] = op.getTxAperture()[ach];
            rxApertures[dstModule][opNumber][dstChannel] = op.getRxAperture()[ach];
            txDelaysList[dstModule][opNumber][dstChannel] = op.getTxDelays()[ach];

            // FC Mapping stuff
            if(op.getRxAperture()[ach]) {
                isRxNop = false;
                frameModule(frameNumber, activeAdapterCh) = dstModule;
                frameModule(frameNumber, activeAdapterCh) = activeUs4oemCh[dstModule];
                ++activeAdapterCh;
                ++activeUs4oemCh[dstModule];
            }
        }
        if(!isRxNop) {
            ++frameNumber;
        }
        ++opNumber;
    }

    // Create operations for each of the us4oem module.
    std::vector<TxRxParamsSequence> seqs(us4oems.size());

    Ordinal us4oemOrdinal = 0;
    for(auto &us4oem : us4oems) {
        auto &us4oemSeq = seqs[us4oemOrdinal];

        uint16 i = 0;
        for(const auto &op : seq) {
            // Convert tx aperture to us4oem tx aperture
            const auto &txAperture = txApertures[us4oemOrdinal][i];
            const auto &rxAperture = rxApertures[us4oemOrdinal][i];
            const auto &txDelays = txDelaysList[us4oemOrdinal][i];

            us4oemSeq.emplace_back(txAperture, txDelays, op.getTxPulse(),
                                   rxAperture, op.getRxSampleRange(),
                                   op.getRxDecimationFactor(), op.getPri());
            ++i;
        }
        // keep operations with empty tx or rx aperture - they are still a part of the larger operation
        ++us4oemOrdinal;
    }
    // split operations if necessary

    auto[splittedOps, opDestSplittedOp, opDestSplittedCh] = splitRxAperturesIfNecessary(seqs);

    // set sequence on each us4oem
    us4oemOrdinal = 0;
    std::vector<FrameChannelMapping::Handle> fcMappings;
    int32 totalNumberOfFrames = 0;
    std::vector<FrameChannelMapping::FrameNumber> frameOffsets(seqs.size(), 0);
    for(auto &us4oem : us4oems) {
        auto fcMapping = us4oem->setTxRxSequence(splittedOps[us4oemOrdinal], tgcSamples);
        frameOffsets[us4oemOrdinal] = totalNumberOfFrames;
        totalNumberOfFrames += fcMapping->getNumberOfFrames();
        fcMappings.push_back(fcMapping);
        ++us4oemOrdinal;
    }

    // generate FrameChannelMapping for the adapter output.
    FrameChannelMappingBuilder outFcBuilder(totalNumberOfFrames, rxApertureSize);
    size_t frameIdx = 0;
    for(const auto &op: seq) {
        if(op.isRxNOP()) {
            continue;
        }

        size_t activeRxChIdx = 0;
        for(auto bit : op.getRxAperture()) {
            if(bit) {
                auto destModule = frameModule(frameIdx, activeRxChIdx);
                auto destModuleChannel = frameChannel(frameIdx, activeRxChIdx);
                // both should be non-negative

                auto destOp = opDestSplittedOp(destModule, frameIdx, destModuleChannel);
                auto destChannel = opDestSplittedCh(destModule, frameIdx, destModuleChannel);

                auto[destFrame, destFrameChannel] = fcMappings[destModule]->getChannel(destOp, destChannel);
                // TODO offset for frames us4oem:1, etc.

                outFcBuilder.setChannelMapping(
                    frameIdx, activeRxChIdx,
                    destFrame + frameOffsets[destModule],
                    destFrameChannel);

                ++activeRxChIdx;
            }
        }
        ++frameIdx;
    }
    return outFcBuilder.build();
}

}