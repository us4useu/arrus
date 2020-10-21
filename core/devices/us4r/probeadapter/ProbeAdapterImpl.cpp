#include "ProbeAdapterImpl.h"

#include "arrus/core/external/eigen/Dense.h"
#include "arrus/core/devices/us4r/common.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/common/aperture.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace arrus::devices {

using namespace ::arrus::ops::us4r;

ProbeAdapterImpl::ProbeAdapterImpl(DeviceId deviceId,
                                   ProbeAdapterModelId modelId,
                                   std::vector<Us4OEMImplBase::RawHandle> us4oems,
                                   ChannelIdx numberOfChannels,
                                   ChannelMapping channelMapping)
    : ProbeAdapterImplBase(deviceId), logger(getLoggerFactory()->getLogger()),
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
        ARRUS_VALIDATOR_EXPECT_IN_RANGE(txRxs.size(), size_t(1), size_t(2048));
        throwOnErrors();

        const auto nSamples = txRxs[0].getNumberOfSamples();
        size_t nActiveRxChannels = std::accumulate(std::begin(txRxs[0].getRxAperture()),
                                                   std::end(txRxs[0].getRxAperture()), 0);
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            auto firingStr = ::arrus::format("firing {}", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getRxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxDelays().size(), size_t(nChannels), firingStr);

            ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getNumberOfSamples() == nSamples,
                                          "Each Rx should acquire the same number of samples.");
            size_t currActiveRxChannels = std::accumulate(std::begin(txRxs[0].getRxAperture()),
                                                          std::end(txRxs[0].getRxAperture()), 0);
            ARRUS_VALIDATOR_EXPECT_TRUE_M(currActiveRxChannels == nActiveRxChannels,
                                          "Each rx aperture should have the same size.");
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

    // -- Frame channel mapping stuff related to splitting each operation between available
    // modules.

    // (logical frame, logical channel) -> physical module
    // active rx channel number here refers to the LOCAL ordinal number of
    // active channel in the rx aperture; e.g. for rx aperture [0, 32, 42],
    // 0 has relative ordinal number 0, 32 has relative number 1,
    // 42 has relative number 2.
    Eigen::MatrixXi frameModule(nFrames, rxApertureSize);
    frameModule.setConstant(FrameChannelMapping::UNAVAILABLE);
    // (logical frame, logical channel) -> actual channel on a given us4oem
    Eigen::MatrixXi frameChannel(nFrames, rxApertureSize);
    frameChannel.setConstant(FrameChannelMapping::UNAVAILABLE);

    // Initialize helper arrays.
    for(Ordinal ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
        txApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        rxApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        txDelaysList.emplace(ordinal, std::vector<std::vector<float>>(seq.size()));
    }

    // Split Tx, Rx apertures and tx delays into sub-apertures specific for
    // each us4oem module.
    uint32 opNumber = 0;

    uint32 frameNumber = 0;
    for(const auto &op : seq) {
        logger->log(LogSeverity::TRACE, arrus::format("Setting tx/rx {}", ::arrus::toString(op)));

        const auto &txAperture = op.getTxAperture();
        const auto &rxAperture = op.getRxAperture();
        const auto &txDelays = op.getTxDelays();
        ARRUS_REQUIRES_TRUE(txAperture.size() == rxAperture.size()
                            && txAperture.size() == numberOfChannels,
                            arrus::format(
                                "Tx and Rx apertures should have a size: {}",
                                numberOfChannels));

        for(Ordinal ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
            txApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
            rxApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
            txDelaysList[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
        }

        size_t activeAdapterCh = 0;
        bool isRxNop = true;
        std::vector<size_t> activeUs4oemCh(us4oems.size(), 0);

        // SPLIT tx/rx/delays between modules
        for(size_t ach = 0; ach < numberOfChannels; ++ach) {

            // tx/rx/delays mapping stuff
            auto cm = channelMapping[ach];
            Ordinal dstModule = cm.first;
            ChannelIdx dstChannel = cm.second;
            txApertures[dstModule][opNumber][dstChannel] = txAperture[ach];
            rxApertures[dstModule][opNumber][dstChannel] = rxAperture[ach];
            txDelaysList[dstModule][opNumber][dstChannel] = txDelays[ach];

            // FC Mapping stuff
            if(op.getRxAperture()[ach]) {
                isRxNop = false;
                frameModule(frameNumber, activeAdapterCh) = dstModule;
                frameChannel(frameNumber, activeAdapterCh) =
                    static_cast<int32>(activeUs4oemCh[dstModule]);
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

    for(Ordinal us4oemOrdinal = 0; us4oemOrdinal < us4oems.size(); ++us4oemOrdinal) {
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
    }
    // split operations if necessary

    auto[splittedOps, opDestSplittedOp, opDestSplittedCh] = splitRxAperturesIfNecessary(seqs);

    // set sequence on each us4oem
    std::vector<FrameChannelMapping::Handle> fcMappings;
    FrameChannelMapping::FrameNumber totalNumberOfFrames = 0;
    std::vector<FrameChannelMapping::FrameNumber> frameOffsets(seqs.size(), 0);

    for(Ordinal us4oemOrdinal = 0; us4oemOrdinal < us4oems.size(); ++us4oemOrdinal) {
        auto& us4oem = us4oems[us4oemOrdinal];
        auto fcMapping = us4oem->setTxRxSequence(splittedOps[us4oemOrdinal], tgcSamples);
        frameOffsets[us4oemOrdinal] = totalNumberOfFrames;
        totalNumberOfFrames += fcMapping->getNumberOfLogicalFrames();
        fcMappings.push_back(std::move(fcMapping));
        // fcMapping is not valid anymore here
    }

    // generate FrameChannelMapping for the adapter output.
    FrameChannelMappingBuilder outFcBuilder(nFrames, rxApertureSize);
    FrameChannelMappingBuilder::FrameNumber frameIdx = 0;
    for(const auto &op: seq) {
        if(op.isRxNOP()) {
            continue;
        }

        uint16 activeRxChIdx = 0;
        for(auto bit : op.getRxAperture()) {
            if(bit) {
                // Frame channel mapping determined by distributing op on multiple devices
                auto dstModule = frameModule(frameIdx, activeRxChIdx);
                auto dstModuleChannel = frameChannel(frameIdx, activeRxChIdx);
                ARRUS_REQUIRES_TRUE_E(
                    dstModule >= 0 && dstModuleChannel >= 0,
                    arrus::ArrusException("Dst module and dst channel "
                                          "should be non-negative")
                );

                auto destOp = opDestSplittedOp(dstModule, frameIdx, dstModuleChannel);
                auto destChannel = opDestSplittedCh(dstModule, frameIdx, dstModuleChannel);
                FrameChannelMapping::FrameNumber destFrame = 0;
                int8 destFrameChannel = -1;
                if(!FrameChannelMapping::isChannelUnavailable(destChannel)) {
                    auto res = fcMappings[dstModule]->getLogical(destOp, destChannel);
                    destFrame = res.first;
                    destFrameChannel = res.second;
                }
                outFcBuilder.setChannelMapping(
                    frameIdx, activeRxChIdx,
                    destFrame + frameOffsets[dstModule],
                    destFrameChannel);

                ++activeRxChIdx;
            }
        }
        ++frameIdx;
    }
    return outFcBuilder.build();
}

}