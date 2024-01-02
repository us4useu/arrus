#include "ProbeAdapterImpl.h"

#include <thread>
#include "arrus/core/external/eigen/Dense.h"
#include "arrus/core/devices/us4r/common.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/common/aperture.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/common/utils.h"

#undef ERROR

namespace arrus::devices {

using namespace ::arrus::ops::us4r;
using ::arrus::ops::us4r::Scheme;


const TxRxParameters &getFirstProbeAdapterRxOp(const std::vector<TxRxParameters> &seq) {
    for(auto &op: seq) {
        if(! op.isRxNOP()) {
            return op;
        }
    }
    throw IllegalArgumentException("All Rxs have empty aperture, this is not allowed.");
}


size_t getProbeAdapterRxApertureSize(const std::vector<TxRxParameters> &seq) {
    auto const &op = getFirstProbeAdapterRxOp(seq);
    auto paddingSize = op.getRxPadding().sum();
    size_t rxApertureSize = getNumberOfActiveChannels(op.getRxAperture()) + paddingSize;
    ARRUS_REQUIRES_TRUE(rxApertureSize > 0, "At least one TX/RX should have non-empty RX aperture.");
    return rxApertureSize;
}


ProbeAdapterImpl::ProbeAdapterImpl(
    DeviceId deviceId, ProbeAdapterModelId modelId,
    std::vector<Us4OEMImplBase::RawHandle> us4oems, ChannelIdx numberOfChannels,
    ChannelMapping channelMapping,
    const ::arrus::devices::us4r::IOSettings &ioSettings)
        : ProbeAdapterImplBase(deviceId), logger(getLoggerFactory()->getLogger()),
          modelId(std::move(modelId)),
          us4oems(std::move(us4oems)),
          numberOfChannels(numberOfChannels),
          channelMapping(std::move(channelMapping)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    frameMetadataOem = getFrameMetadataOem(ioSettings);
}

class ProbeAdapterTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    ProbeAdapterTxRxValidator(const std::string &componentName, ChannelIdx nChannels)
            : Validator(componentName), nChannels(nChannels) {}

    void validate(const TxRxParamsSequence &txRxs) override {
        auto const &refOp = getFirstProbeAdapterRxOp(txRxs);
        auto nSamples = refOp.getNumberOfSamples();
        size_t nActiveRxChannels = std::accumulate(std::begin(refOp.getRxAperture()), std::end(refOp.getRxAperture()), 0);
        nActiveRxChannels += refOp.getRxPadding().sum();
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            auto firingStr = ::arrus::format("firing {}", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getRxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxDelays().size(), size_t(nChannels), firingStr);

            if(!op.isRxNOP()) {
                ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getNumberOfSamples() == nSamples,
                                              "Each Rx should acquire the same number of samples.");
                size_t currActiveRxChannels = std::accumulate(std::begin(txRxs[firing].getRxAperture()),
                                                              std::end(txRxs[firing].getRxAperture()), 0);
                currActiveRxChannels += txRxs[firing].getRxPadding().sum();
                ARRUS_VALIDATOR_EXPECT_TRUE_M(currActiveRxChannels == nActiveRxChannels,
                                              "Each rx aperture should have the same size.");
            }
            if(hasErrors()) {
                return;
            }
        }
    }
private:
    ChannelIdx nChannels;
};


std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
ProbeAdapterImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples,
                                  uint16 rxBufferSize, uint16 batchSize, std::optional<float> sri, bool triggerSync,
                                  const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc,
                                  const std::vector<arrus::framework::NdArray> &txDelayProfiles) {
    // Validate input sequence
    ProbeAdapterTxRxValidator validator(::arrus::format("{} tx rx sequence", getDeviceId().toString()), numberOfChannels);
    validator.validate(seq);
    validator.throwOnErrors();

    // Split into multiple arrays.
    // us4oem, op number -> aperture/delays
    std::unordered_map<Ordinal, std::vector<BitMask>> txApertures, rxApertures;
    std::unordered_map<Ordinal, std::vector<std::vector<float>>> txDelaysList;
    std::unordered_map<Ordinal, std::vector<arrus::framework::NdArray>> txDelayProfilesList;

    // Here is an assumption, that each operation has the same size rx aperture, except RX nops.
    auto nFrames = getNumberOfNoRxNOPs(seq);
    // find the first non rx NOP and use it to determine rxApertureSize
    size_t rxApertureSize = getProbeAdapterRxApertureSize(seq);

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

    ::arrus::framework::NdArray::Shape txDelaysProfileShape = {seq.size(), Us4OEMImpl::N_TX_CHANNELS};

    // Initialize helper arrays.
    for(Ordinal ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
        txApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        rxApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        txDelaysList.emplace(ordinal, std::vector<std::vector<float>>(seq.size()));

        // Profiles.
        std::vector<arrus::framework::NdArray> txDelayProfilesForModule;
        size_t nProfiles = txDelayProfiles.size();
        for(size_t i = 0; i < nProfiles; ++i) {
            ::arrus::framework::NdArray emptyArray(
                txDelaysProfileShape,
                txDelayProfiles[i].getDataType(),
                txDelayProfiles[i].getPlacement(),
                txDelayProfiles[i].getName()
            );
            txDelayProfilesForModule.push_back(std::move(emptyArray));
        }
        txDelayProfilesList.emplace(ordinal, txDelayProfilesForModule);
    }

    // Split Tx, Rx apertures and tx delays into sub-apertures specific for each us4oem module.
    uint32 opNumber = 0;
    uint32 frameNumber = 0;
    for(const auto &op : seq) {
        logger->log(LogSeverity::TRACE, arrus::format("Setting tx/rx {}", ::arrus::toString(op)));
        const auto &txAperture = op.getTxAperture();
        const auto &rxAperture = op.getRxAperture();
        const auto &txDelays = op.getTxDelays();

        std::vector<std::vector<int32>> us4oemChannels(us4oems.size());
        std::vector<std::vector<int32>> adapterChannels(us4oems.size());

        ARRUS_REQUIRES_TRUE(txAperture.size() == rxAperture.size() && txAperture.size() == numberOfChannels,
                            format("Tx and Rx apertures should have a size: {}", numberOfChannels));
        for (Ordinal ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
            txApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS, false);
            rxApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS, false);
            txDelaysList[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS, 0.0f);
        }
        size_t activeAdapterCh = 0;
        bool isRxNop = true;

        // SPLIT tx/rx/delays between modules
        for(size_t ach = 0; ach < numberOfChannels; ++ach) {
            // tx/rx/delays mapping stuff
            auto cm = channelMapping[ach];
            Ordinal dstModule = cm.first;
            ChannelIdx dstChannel = cm.second;
            txApertures[dstModule][opNumber][dstChannel] = txAperture[ach];
            rxApertures[dstModule][opNumber][dstChannel] = rxAperture[ach];
            txDelaysList[dstModule][opNumber][dstChannel] = txDelays[ach];

            for(size_t i = 0; i < txDelayProfiles.size(); ++i) {
                txDelayProfilesList[dstModule][i].set(opNumber, dstChannel, txDelayProfiles[i].get<float>(opNumber, ach));
            }

            // FC Mapping stuff
            if(op.getRxAperture()[ach]) {
                isRxNop = false;
                frameModule(frameNumber, activeAdapterCh+op.getRxPadding()[0]) = dstModule;
                // This will be processed further later.
                us4oemChannels[dstModule].push_back(static_cast<int32>(dstChannel));
                adapterChannels[dstModule].push_back(static_cast<int32>(activeAdapterCh+op.getRxPadding()[0]));
                ++activeAdapterCh;
            }
        }
        if(!isRxNop) {
            // FCM
            // Compute rank of each us4oem RX channel (to get the "aperture" channel number).
            // The rank is needed, as the further code decomposes each op into 32-rx element ops
            // assuming, that the first 32 channels of rx aperture will be used in the first
            // op, the next 32 channels in the second op and so on.
            for(Ordinal ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
                auto &uChannels = us4oemChannels[ordinal];
                auto &aChannels = adapterChannels[ordinal];
                auto rxApertureChannels = ::arrus::rank(uChannels);
                for(size_t c = 0; c < uChannels.size(); ++c) {
                    frameChannel(frameNumber, aChannels[c]) = static_cast<int32>(rxApertureChannels[c]);
                }
            }
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

            // Intentionally not copying rx padding - us4oem do not allow rx padding.
            us4oemSeq.emplace_back(txAperture, txDelays, op.getTxPulse(),
                                   rxAperture, op.getRxSampleRange(),
                                   op.getRxDecimationFactor(), op.getPri(),
                                   Tuple<ChannelIdx>({0, 0}),
                                   op.getRxDelay(),
                                   op.getBitstreamId()
                                   );
            ++i;
        }
        // keep operations with empty tx or rx aperture - they are still a part of the larger operation
    }
    // creating 32-element subapertures
    std::vector<std::vector<uint8_t>> us4oemL2PChannelMappings;
    for(auto &us4oem: us4oems) {
        us4oemL2PChannelMappings.push_back(us4oem->getChannelMapping());
    }

    auto[splittedOps, opDstSplittedOp, opDestSplittedCh, us4oemTxDelayProfiles] = splitRxAperturesIfNecessary(
        seqs, us4oemL2PChannelMappings, txDelayProfilesList, frameMetadataOem);

    calculateRxDelays(splittedOps);

    // set sequence on each us4oem
    std::vector<FrameChannelMapping::Handle> fcMappings;
    // section -> us4oem -> transfer
    std::vector<std::vector<DataTransfer>> outputTransfers;
    uint32 currentFrameOffset = 0;
    std::vector<uint32> frameOffsets(static_cast<unsigned int>(us4oems.size()), 0);
    std::vector<uint32> numberOfFrames(static_cast<unsigned int>(us4oems.size()), 0);

    Us4RBufferBuilder us4RBufferBuilder;
    for(Ordinal us4oemOrdinal = 0; us4oemOrdinal < us4oems.size(); ++us4oemOrdinal) {
        auto &us4oem = us4oems[us4oemOrdinal];
        std::vector<arrus::framework::NdArray> profile;
        if(!us4oemTxDelayProfiles.empty()) {
            profile = us4oemTxDelayProfiles.at(us4oemOrdinal);
        }
        auto[buffer, fcMapping] = us4oem->upload(splittedOps[us4oemOrdinal], tgcSamples, rxBufferSize,
            batchSize, sri, triggerSync, ddc, profile);
        frameOffsets[us4oemOrdinal] = currentFrameOffset;
        currentFrameOffset += fcMapping->getNumberOfLogicalFrames()*batchSize;
        numberOfFrames[us4oemOrdinal] = fcMapping->getNumberOfLogicalFrames()*batchSize;
        fcMappings.push_back(std::move(fcMapping));
        // fcMapping is not valid anymore here
        us4RBufferBuilder.pushBack(buffer);
    }

    // generate FrameChannelMapping for the adapter output.
    FrameChannelMappingBuilder outFcBuilder(nFrames, ARRUS_SAFE_CAST(rxApertureSize, ChannelIdx));
    FrameChannelMappingBuilder::FrameNumber frameIdx = 0;
    for(const auto &op: seq) {
        if(op.isRxNOP()) {
            continue;
        }
        uint16 activeRxChIdx = 0;
        for(auto bit : op.getRxAperture()) {
            if(bit) {
                // Frame channel mapping determined by distributing op on multiple devices
                auto dstModule = frameModule(frameIdx, activeRxChIdx + op.getRxPadding()[0]);
                auto dstModuleChannel = frameChannel(frameIdx, activeRxChIdx + op.getRxPadding()[0]);

                // if dstModuleChannel is unavailable, set channel mapping to -1 and continue
                // unavailable dstModuleChannel means, that the given channel was virtual
                // and has no assigned value.
                ARRUS_REQUIRES_DATA_TYPE_E(dstModuleChannel, int8,ArrusException("Invalid dstModuleChannel data type"));
                if(FrameChannelMapping::isChannelUnavailable((int8) dstModuleChannel)) {
                    outFcBuilder.setChannelMapping(frameIdx, activeRxChIdx + op.getRxPadding()[0],
                                                   0, 0, FrameChannelMapping::UNAVAILABLE);
                } else {
                    // Otherwise, we have an actual channel.
                    ARRUS_REQUIRES_TRUE_E(dstModule >= 0 && dstModuleChannel >= 0,
                                          ArrusException("Dst module and dst channel should be non-negative"));

                    // dstOp, dstChannel - frame and channel after considering that the aperture ops are
                    // into multiple smaller ops for each us4oem separately.
                    // dstOp, dstChannel - frame and channel of a given module
                    auto dstOp = opDstSplittedOp(dstModule, frameIdx, dstModuleChannel);
                    auto dstChannel = opDestSplittedCh(dstModule, frameIdx, dstModuleChannel);
                    FrameChannelMapping::Us4OEMNumber us4oem = 0;
                    FrameChannelMapping::FrameNumber dstFrame = 0;
                    int8 dstFrameChannel = -1;
                    if(!FrameChannelMapping::isChannelUnavailable(dstChannel)) {
                        auto res = fcMappings[dstModule]->getLogical(dstOp, dstChannel);
                        us4oem = arrus::devices::get<0>(res);
                        dstFrame = arrus::devices::get<1>(res);
                        dstFrameChannel = arrus::devices::get<2>(res);
                    }
                    outFcBuilder.setChannelMapping(frameIdx, activeRxChIdx + op.getRxPadding()[0],
                                                   us4oem, dstFrame, dstFrameChannel);
                }
                ++activeRxChIdx;
            }
        }
        ++frameIdx;
    }
    outFcBuilder.setFrameOffsets(frameOffsets);
    outFcBuilder.setNumberOfFrames(numberOfFrames);
    return {us4RBufferBuilder.build(), outFcBuilder.build()};
}

Ordinal ProbeAdapterImpl::getNumberOfUs4OEMs() {
    return ARRUS_SAFE_CAST(this->us4oems.size(), Ordinal);
}

void ProbeAdapterImpl::start() {
//  EnableSequencer resets position of the us4oem sequencer.
    for(auto &us4oem: this->us4oems) {
        us4oem->getIUs4oem()->DisableWaitOnReceiveOverflow();
        us4oem->getIUs4oem()->DisableWaitOnTransferOverflow();
        // Reset tx subsystem pointers.
        us4oem->getIUs4oem()->EnableTransmit();
        // Reset sequencer pointers.
        us4oem->enableSequencer();
    }
    this->us4oems[0]->startTrigger();
}

void ProbeAdapterImpl::stop() {
    this->us4oems[0]->stop();
}

void ProbeAdapterImpl::syncTrigger() {
    this->us4oems[0]->syncTrigger();
}
Ordinal ProbeAdapterImpl::getFrameMetadataOem(const us4r::IOSettings &settings) {
    if(!settings.hasFrameMetadataCapability()) {
        return 0; // By default us4OEM:0 is considered to provide frame metadata
    }
    else {
        std::unordered_set<Ordinal> oems = settings.getFrameMetadataCapabilityOEMs();
        if(oems.size() != 1) {
            throw ::arrus::IllegalArgumentException("Exactly one OEM should be set for the pulse counter capability.");
        }
        else {
            // Only a single OEM.
            return *std::begin(oems);
        }
    }
}

/**
 * NOTE: this method works in-place (modifies input sequence).
 */
void ProbeAdapterImpl::calculateRxDelays(std::vector<TxRxParamsSequence> &sequences) {
    auto nUs4OEMs = sequences.size();
    auto sequenceSize = sequences[0].size();
    for(size_t txrx = 0; txrx < sequenceSize; ++txrx) {
        float maxDelay = 0.0f;
        for(size_t oem = 0; oem < nUs4OEMs; ++oem) {
            TxRxParameters &op = sequences[oem][txrx];
            std::vector<float> delays;
            // NOTE: assuming that TX aperture and delays have the same length.
            // Filtering out tx delay values that have tx delay == 0.
            for(size_t i = 0; i < op.getTxAperture().size(); ++i) {
                if(op.getTxAperture()[i]) {
                    delays.push_back(op.getTxDelays()[i]);
                }
            }
            if(!delays.empty()) {
                // TX delay
                float txrxMaxDelay = *std::max_element(std::begin(delays), std::end(delays));
                // burst time
                float frequency = op.getTxPulse().getCenterFrequency();
                float nPeriods = op.getTxPulse().getNPeriods();
                float burstTime = 1.0f/frequency*nPeriods;
                // Total rx delay
                float newDelay = txrxMaxDelay + burstTime;
                if(newDelay > maxDelay) {
                    maxDelay = newDelay;
                }
            }
        }
        // Set Rx delays in the input sequences.
        for(size_t oem = 0; oem < nUs4OEMs; ++oem) {
            sequences[oem][txrx].setRxDelay(maxDelay);
        }
    }
}

}
