#include "ProbeAdapterImpl.h"

#include "arrus/core/external/eigen/Dense.h"
#include "arrus/core/devices/us4r/common.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/common/aperture.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/common/utils.h"

#undef ERROR

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
        const auto nSamples = txRxs[0].getNumberOfSamples();
        size_t nActiveRxChannels = std::accumulate(std::begin(txRxs[0].getRxAperture()),
                                                   std::end(txRxs[0].getRxAperture()), 0)
                                   + txRxs[0].getRxPadding().sum();
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
            size_t currActiveRxChannels = std::accumulate(std::begin(txRxs[firing].getRxAperture()),
                                                          std::end(txRxs[firing].getRxAperture()), 0)
                                          + txRxs[firing].getRxPadding().sum();
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
ProbeAdapterImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                                  const TGCCurve &tgcSamples,
                                  uint16 rxBufferSize, uint16 batchSize,
                                  std::optional<float> sri) {
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

    // Split Tx, Rx apertures and tx delays into sub-apertures specific for
    // each us4oem module.
    uint32 opNumber = 0;

    uint32 frameNumber = 0;
    for(const auto &op : seq) {
        logger->log(LogSeverity::TRACE, arrus::format("Setting tx/rx {}", ::arrus::toString(op)));

        const auto &txAperture = op.getTxAperture();
        const auto &rxAperture = op.getRxAperture();
        const auto &txDelays = op.getTxDelays();

        // TODO change the below to an 'assert'
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
                frameModule(frameNumber, activeAdapterCh + op.getRxPadding()[0]) = dstModule;
                frameChannel(frameNumber, activeAdapterCh + op.getRxPadding()[0]) =
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

    auto[splittedOps, opDstSplittedOp, opDestSplittedCh] = splitRxAperturesIfNecessary(seqs);

    // set sequence on each us4oem
    std::vector<FrameChannelMapping::Handle> fcMappings;
    FrameChannelMapping::FrameNumber totalNumberOfFrames = 0;
    std::vector<FrameChannelMapping::FrameNumber> frameOffsets(seqs.size(), 0);

    // section -> us4oem -> transfer
    std::vector<std::vector<DataTransfer>> outputTransfers;

    Us4RBufferBuilder us4RBufferBuilder;
    for(Ordinal us4oemOrdinal = 0; us4oemOrdinal < us4oems.size(); ++us4oemOrdinal) {
        auto &us4oem = us4oems[us4oemOrdinal];
        auto[buffer, fcMapping] = us4oem->setTxRxSequence(
            splittedOps[us4oemOrdinal], tgcSamples, rxBufferSize, batchSize,
            sri);
        frameOffsets[us4oemOrdinal] = totalNumberOfFrames;
        totalNumberOfFrames += fcMapping->getNumberOfLogicalFrames();
        fcMappings.push_back(std::move(fcMapping));
        // fcMapping is not valid anymore here
        us4RBufferBuilder.pushBackUs4oemBuffer(buffer);
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
                ARRUS_REQUIRES_DATA_TYPE_E(
                    dstModuleChannel, int8,
                    ::arrus::ArrusException(
                        "Invalid dstModuleChannel data type, "
                        "rx aperture is outise."));
                if(FrameChannelMapping::isChannelUnavailable((int8) dstModuleChannel)) {
                    outFcBuilder.setChannelMapping(
                        frameIdx, activeRxChIdx + op.getRxPadding()[0],
                        0, FrameChannelMapping::UNAVAILABLE);
                } else {
                    // Otherwise, we have an actual channel.
                    ARRUS_REQUIRES_TRUE_E(
                        dstModule >= 0 && dstModuleChannel >= 0,
                        arrus::ArrusException("Dst module and dst channel "
                                              "should be non-negative")
                    );

                    auto dstOp = opDstSplittedOp(dstModule, frameIdx, dstModuleChannel);
                    auto dstChannel = opDestSplittedCh(dstModule, frameIdx, dstModuleChannel);
                    FrameChannelMapping::FrameNumber destFrame = 0;
                    int8 destFrameChannel = -1;
                    if(!FrameChannelMapping::isChannelUnavailable(dstChannel)) {
                        auto res = fcMappings[dstModule]->getLogical(dstOp, dstChannel);
                        destFrame = res.first;
                        destFrameChannel = res.second;
                    }
                    outFcBuilder.setChannelMapping(
                        frameIdx, activeRxChIdx + op.getRxPadding()[0],
                        destFrame + frameOffsets[dstModule],
                        destFrameChannel);
                }
                ++activeRxChIdx;
            }
        }
        ++frameIdx;
    }
    return {us4RBufferBuilder.build(), outFcBuilder.build()};
}

Ordinal ProbeAdapterImpl::getNumberOfUs4OEMs() {
    return ARRUS_SAFE_CAST(this->us4oems.size(), Ordinal);
}

void ProbeAdapterImpl::start() {
//     EnableSequencer resets position of the us4oem sequencer.
    for(auto &us4oem: this->us4oems) {
        us4oem->enableSequencer();
    }
    this->us4oems[0]->start();
}

void ProbeAdapterImpl::stop() {
    this->us4oems[0]->stop();
}

void ProbeAdapterImpl::syncTrigger() {
    this->us4oems[0]->syncTrigger();
}

void ProbeAdapterImpl::registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &transfers) {
    Ordinal us4oemOrdinal = 0;
    for(auto &us4oem: us4oems) {
        auto us4oemBuffer = transfers->getUs4oemBuffer(us4oemOrdinal);
        registerOutputBuffer(buffer, us4oemBuffer, us4oem);
        ++us4oemOrdinal;
    }
}

void ProbeAdapterImpl::registerOutputBuffer(Us4ROutputBuffer *outputBuffer,
                                            const Us4OEMBuffer &transfers,
                                            Us4OEMImplBase::RawHandle us4oem) {
    // Each transfer should have the same size.
    std::unordered_set<size_t> sizes;
    for(auto &transfer: transfers.getElements()) {
        sizes.insert(transfer.getSize());
    }
    if(sizes.size() > 1) {
        throw ::arrus::ArrusException("Each transfer should have the same size.");
    }
    // This is the size of a single element produced by this us4oem.
    const size_t elementSize = *std::begin(sizes);
    if(elementSize == 0) {
        // This us4oem will not transfer any data, so the buffer registration has no sense here.
        return;
    }
    // Output buffer - assuming that the number of elements is a multiple of number of transfers
    const auto rxBufferSize = ARRUS_SAFE_CAST(transfers.getNumberOfElements(), uint16);
    const uint16 hostBufferSize = outputBuffer->getNumberOfElements();
    const Ordinal ordinal = us4oem->getDeviceId().getOrdinal();

    // Prepare host buffers
    uint16 hostElement = 0;
    uint16 rxElement = 0;

    auto ius4oem = us4oem->getIUs4oem();

    while(hostElement < hostBufferSize) {
        auto dstAddress = outputBuffer->getAddress(hostElement, ordinal);
        auto srcAddress = transfers.getElement(rxElement).getAddress();
        logger->log(LogSeverity::DEBUG, ::arrus::format("Preparing host buffer to {} from {}, size {}",
                                                        (size_t) dstAddress, (size_t) srcAddress, elementSize));
        ius4oem->PrepareHostBuffer(dstAddress, elementSize, srcAddress);
        ++hostElement;
        rxElement = (rxElement + 1) % rxBufferSize;
    }

    // prepare transfers
    uint16 transferIdx = 0;
    uint16 startFiring = 0;

    for(auto &transfer: transfers.getElements()) {
        auto dstAddress = outputBuffer->getAddress(transferIdx, ordinal);
        auto srcAddress = transfer.getAddress();
        auto endFiring = transfer.getFiring();

        ius4oem->PrepareTransferRXBufferToHost(
            transferIdx, dstAddress, elementSize, srcAddress);

        ius4oem->ScheduleTransferRXBufferToHost(
            endFiring, transferIdx,
            [this, ius4oem, outputBuffer, ordinal, transferIdx, startFiring,
                endFiring, srcAddress, elementSize,
                rxBufferSize, hostBufferSize,
                element = transferIdx]() mutable {
                try {
                    auto dstAddress = outputBuffer->getAddress((uint16) element, ordinal);
                    ius4oem->MarkEntriesAsReadyForReceive(startFiring, endFiring);
                    logger->log(LogSeverity::DEBUG, ::arrus::format("Rx Released: {}, {}", startFiring, endFiring));

                    // Prepare transfer for the next iteration.
                    ius4oem->PrepareTransferRXBufferToHost(
                        transferIdx, dstAddress, elementSize, srcAddress);
                    ius4oem->ScheduleTransferRXBufferToHost(endFiring, transferIdx, nullptr);

                    bool cont = outputBuffer->signal(ordinal, element,
                                                     HostBuffer::INF_TIMEOUT); // Also a callback function can be used here.
                    if(!cont) {
                        logger->log(LogSeverity::DEBUG, "Output buffer shut down.");
                        return;
                    }
                    // TODO the below should be called when buffer element is released.
                    cont = outputBuffer->waitForRelease(ordinal, element, HostBuffer::INF_TIMEOUT);
                    if(!cont) {
                        logger->log(LogSeverity::DEBUG, "Output buffer shut down");
                        return;
                    }
                    ius4oem->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
                    logger->log(LogSeverity::DEBUG, ::arrus::format("Host Released: {}, {}", startFiring, endFiring));
                    element = (element + rxBufferSize) % hostBufferSize;
                } catch(const std::exception &e) {
                    logger->log(LogSeverity::ERROR, "Us4OEM: "
                                                    + std::to_string(ordinal) +
                                                    " transfer callback ended with an exception: " +
                                                    e.what());
                } catch(...) {
                    logger->log(LogSeverity::ERROR, "Us4OEM: "
                                                    + std::to_string(ordinal) +
                                                    " transfer callback ended with unknown exception");
                   }

            }
        );
        // TODO register in the us4oem
        startFiring = endFiring + 1;
        ++transferIdx;
    }
    // Register overflow callbacks (mark output buffer as invalid)

    ius4oem->RegisterReceiveOverflowCallback([this, outputBuffer]() {
        try {
            this->logger->log(LogSeverity::ERROR, "Rx buffer overflow, stopping the device.");
            this->getMasterUs4oem()->stop();
            outputBuffer->markAsInvalid();
        } catch (const std::exception &e) {
            logger->log(LogSeverity::ERROR, "Receive overflow callback ended with an exception: " +
                                            std::string(e.what()));
        } catch (...) {
            logger->log(LogSeverity::ERROR, "Receive overflow callback ended with unknown exception");
        }

    });

    ius4oem->RegisterTransferOverflowCallback([this, outputBuffer]() {
        try {
            this->logger->log(LogSeverity::ERROR, "Host buffer overflow, stopping the device.");
            this->getMasterUs4oem()->stop();
            outputBuffer->markAsInvalid();
        } catch (const std::exception &e) {
            logger->log(LogSeverity::ERROR, "Receive overflow callback ended with an exception: " +
                                            std::string(e.what()));
        } catch (...) {
            logger->log(LogSeverity::ERROR, "Receive overflow callback ended with unknown exception");
        }

    });
}

void ProbeAdapterImpl::setTgcCurve(const TGCCurve &curve) {
    for(auto &us4oem: us4oems) {
        us4oem->setTgcCurve(curve);
    }
}

}