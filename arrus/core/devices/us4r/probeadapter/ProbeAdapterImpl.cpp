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

ProbeAdapterImpl::ProbeAdapterImpl(DeviceId deviceId, ProbeAdapterModelId modelId,
                                   std::vector<Us4OEMImplBase::RawHandle> us4oems, ChannelIdx numberOfChannels,
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
        const auto nSamples = txRxs[0].getNumberOfSamples();
        size_t nActiveRxChannels = std::accumulate(std::begin(txRxs[0].getRxAperture()), std::end(txRxs[0].getRxAperture()), 0);
        nActiveRxChannels += txRxs[0].getRxPadding().sum();
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            auto firingStr = ::arrus::format("firing {}", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getRxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxDelays().size(), size_t(nChannels), firingStr);

            ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getNumberOfSamples() == nSamples,
                                          "Each Rx should acquire the same number of samples.");
            size_t currActiveRxChannels = std::accumulate(std::begin(txRxs[firing].getRxAperture()),
                                                          std::end(txRxs[firing].getRxAperture()), 0);
            currActiveRxChannels += txRxs[firing].getRxPadding().sum();
            ARRUS_VALIDATOR_EXPECT_TRUE_M(currActiveRxChannels == nActiveRxChannels,
                                          "Each rx aperture should have the same size.");
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
                                  const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc) {
    // Validate input sequence
    ProbeAdapterTxRxValidator validator(::arrus::format("{} tx rx sequence", getDeviceId().toString()), numberOfChannels);
    validator.validate(seq);
    validator.throwOnErrors();

    // Split into multiple arrays.
    // us4oem, op number -> aperture/delays
    std::unordered_map<Ordinal, std::vector<BitMask>> txApertures, rxApertures;
    std::unordered_map<Ordinal, std::vector<std::vector<float>>> txDelaysList;

    // Here is an assumption, that each operation has the same size rx aperture.
    auto paddingSize = seq[0].getRxPadding().sum();
    auto rxApertureSize = getNumberOfActiveChannels(seq[0].getRxAperture()) + paddingSize;
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

        // TODO change the below to an 'assert'
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

            // Intentionally not copying rx padding - us4oem do not allow rx padding.
            us4oemSeq.emplace_back(txAperture, txDelays, op.getTxPulse(),
                                   rxAperture, op.getRxSampleRange(),
                                   op.getRxDecimationFactor(), op.getPri(),
                                   Tuple<ChannelIdx>({0, 0}));
            ++i;
        }
        // keep operations with empty tx or rx aperture - they are still a part of the larger operation
    }
    // split operations if necessary
    std::vector<std::vector<uint8_t>> us4oemL2PChannelMappings;
    for(auto &us4oem: us4oems) {
        us4oemL2PChannelMappings.push_back(us4oem->getChannelMapping());
    }
    auto[splittedOps, opDstSplittedOp, opDestSplittedCh] = splitRxAperturesIfNecessary(seqs, us4oemL2PChannelMappings);

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
        auto[buffer, fcMapping] = us4oem->setTxRxSequence(splittedOps[us4oemOrdinal], tgcSamples, rxBufferSize,
                                                          batchSize, sri, triggerSync, ddc);
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

void ProbeAdapterImpl::registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &us4rBuffer,
                                            Scheme::WorkMode workMode) {
    Ordinal us4oemOrdinal = 0;

    if(transferRegistrar.size() < us4oems.size()) {
        transferRegistrar.resize(us4oems.size());
    }
    for(auto &us4oem: us4oems) {
        auto us4oemBuffer = us4rBuffer->getUs4oemBuffer(us4oemOrdinal);
        registerOutputBuffer(buffer, us4oemBuffer, us4oem, workMode);
        ++us4oemOrdinal;
    }
}

/**
 * - This function assumes, that the size of output buffer (number of elements)
 *  is a multiple of number of us4oem elements.
 * - this function will not schedule data transfer when the us4oem element size is 0.
 */
void ProbeAdapterImpl::registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc,
                                            Us4OEMImplBase *us4oem, Scheme::WorkMode workMode) {
    auto us4oemOrdinal = us4oem->getDeviceId().getOrdinal();
    auto ius4oem = us4oem->getIUs4oem();
    const auto nElementsSrc = bufferSrc.getNumberOfElements();
    const size_t nElementsDst = bufferDst->getNumberOfElements();

    size_t elementSize = getUniqueUs4OEMBufferElementSize(bufferSrc);

    if (elementSize == 0) {
        return;
    }
    if(transferRegistrar[us4oemOrdinal]) {
        transferRegistrar[us4oemOrdinal].reset();
    }
    transferRegistrar[us4oemOrdinal] = std::make_shared<Us4OEMDataTransferRegistrar>(bufferDst, bufferSrc, us4oem);
    transferRegistrar[us4oemOrdinal]->registerTransfers();

    // Register buffer element release functions.
    bool isTriggerRequired = workMode == Scheme::WorkMode::HOST;
    bool isMaster = us4oem->getDeviceId().getOrdinal() == this->getMasterUs4oem()->getDeviceId().getOrdinal();
    size_t nRepeats = nElementsDst/nElementsSrc;
    uint16 startFiring = 0;

    for(size_t i = 0; i < bufferSrc.getNumberOfElements(); ++i) {
        auto &srcElement = bufferSrc.getElement(i);
        uint16 endFiring = srcElement.getFiring();
        for(size_t j = 0; j < nRepeats; ++j) {
            std::function<void()> releaseFunc;
            if(isTriggerRequired) {
                releaseFunc = [this, startFiring, endFiring]() {
                    for(int i = us4oems.size()-1; i >= 0; --i) {
                        us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForReceive(startFiring, endFiring);
                        us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
                    }
                    getMasterUs4oem()->syncTrigger();
                };
            }
            else {
                releaseFunc = [this, startFiring, endFiring]() {
                    for(int i = us4oems.size()-1; i >= 0; --i) {
                        us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForReceive(startFiring, endFiring);
                        us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
                    }
                };
            }
            bufferDst->registerReleaseFunction(j*nElementsSrc+i, releaseFunc);
        }
        startFiring = endFiring+1;
    }

    // Overflow handling
    using namespace std::chrono_literals;
    ius4oem->RegisterReceiveOverflowCallback([this, bufferDst, isMaster]() {
        try {
            if(bufferDst->isStopOnOverflow()) {
                this->logger->log(LogSeverity::ERROR, "Rx data overflow, stopping the device.");
                size_t nElements = bufferDst->getNumberOfElements();
                while(nElements != bufferDst->getNumberOfElementsInState(framework::BufferElement::State::FREE)) {
                    std::this_thread::sleep_for(1ms);
                }
                if(isMaster) {
                    for(int i = us4oems.size()-1; i >= 0; --i) {
                        us4oems[i]->getIUs4oem()->SyncReceive();
                    }
                }
//                this->getMasterUs4oem()->stop();
//                bufferDst->markAsInvalid();
            } else {
                this->logger->log(LogSeverity::WARNING, "Rx data overflow ...");

            }
        } catch (const std::exception &e) {
            logger->log(LogSeverity::ERROR, format("RX overflow callback exception: ", e.what()));
        } catch (...) {
            logger->log(LogSeverity::ERROR, "RX overflow callback exception: unknown");
        }
    });

    ius4oem->RegisterTransferOverflowCallback([this, bufferDst, isMaster]() {
        try {
            if(bufferDst->isStopOnOverflow()) {
                this->logger->log(LogSeverity::ERROR, "Host data overflow, stopping the device.");
                size_t nElements = bufferDst->getNumberOfElements();
                while(nElements != bufferDst->getNumberOfElementsInState(framework::BufferElement::State::FREE)) {
                    std::this_thread::sleep_for(1ms);
                }
                if(isMaster) {
                    for(int i = us4oems.size()-1; i >= 0; --i) {
                        us4oems[i]->getIUs4oem()->SyncTransfer();
                    }
                }
//                this->getMasterUs4oem()->stop();
//                bufferDst->markAsInvalid();
            }
            else {
                this->logger->log(LogSeverity::WARNING, "Host data overflow ...");
            }
        } catch (const std::exception &e) {
            logger->log(LogSeverity::ERROR, format("Host overflow callback exception: ", e.what()));
        } catch (...) {
            logger->log(LogSeverity::ERROR, "Host overflow callback exception: unknown");
        }
    });
    ius4oem->EnableWaitOnReceiveOverflow();
    ius4oem->EnableWaitOnTransferOverflow();
}

size_t ProbeAdapterImpl::getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const {
    std::unordered_set<size_t> sizes;
    for (auto &element: us4oemBuffer.getElements()) {
        sizes.insert(element.getSize());
    }
    if (sizes.size() > 1) {
        throw ArrusException("Each us4oem buffer element should have the same size.");
    }
    // This is the size of a single element produced by this us4oem.
    const size_t elementSize = *std::begin(sizes);
    return elementSize;
}

void ProbeAdapterImpl::unregisterOutputBuffer() {
    if(transferRegistrar.empty()) {
        return;
    }
    for (Ordinal i = 0; i < us4oems.size(); ++i) {
        if(transferRegistrar[i]) {
            transferRegistrar[i]->unregisterTransfers();
        }
    }
}

}
