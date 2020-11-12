#include "Us4OEMImpl.h"

#include <cmath>
#include <thread>
#include <chrono>

#include "arrus/common/format.h"
#include "arrus/common/utils.h"
#include "arrus/core/common/collections.h"
#include "arrus/common/asserts.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/common/interpolate.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace arrus::devices {

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
                       const BitMask &activeChannelGroups,
                       std::vector<uint8_t> channelMapping,
                       uint16 pgaGain, uint16 lnaGain,
                       std::unordered_set<uint8_t> channelsMask)
    : Us4OEMImplBase(id), logger{getLoggerFactory()->getLogger()},
      ius4oem(std::move(ius4oem)),
      channelMapping(std::move(channelMapping)),
      channelsMask(std::move(channelsMask)),
      pgaGain(pgaGain), lnaGain(lnaGain) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());

    // This class stores reordered active groups of channels,
    // as presented in the IUs4OEM docs.
    static const std::vector<ChannelIdx> acgRemap = {0, 4, 8, 12,
                                                     2, 6, 10, 14,
                                                     1, 5, 9, 13,
                                                     3, 7, 11, 15};
    auto acg = ::arrus::permute(activeChannelGroups, acgRemap);
    ARRUS_REQUIRES_TRUE(acg.size() == activeChannelGroups.size(),
                        arrus::format(
                            "Invalid number of active channels mask elements; "
                            "the input has {}, expected: {}", acg.size(),
                            activeChannelGroups.size()));
    this->activeChannelGroups = ::arrus::toBitset<N_ACTIVE_CHANNEL_GROUPS>(acg);

    if(this->channelsMask.empty()) {
        this->logger->log(LogSeverity::INFO, ::arrus::format(
            "No channel masking will be applied for {}", ::arrus::toString(id)));
    } else {
        this->logger->log(LogSeverity::INFO, ::arrus::format(
            "Following us4oem channels will be turned off: {}",
            ::arrus::toString(this->channelsMask)));
    }
}

Us4OEMImpl::~Us4OEMImpl() {
    try {
        logger->log(LogSeverity::DEBUG, arrus::format("Destroying handle"));
    } catch(const std::exception &e) {
        std::cerr <<
                  arrus::format("Exception while calling us4oem destructor: {}",
                                e.what())
                  << std::endl;
    }
}

bool Us4OEMImpl::isMaster() {
    return getDeviceId().getOrdinal() == 0;
}

void Us4OEMImpl::startTrigger() {
    if(isMaster()) {
        ius4oem->TriggerStart();
    }
}

void Us4OEMImpl::stopTrigger() {
    if(isMaster()) {
        ius4oem->TriggerStop();
    }
}

class Us4OEMTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    using Validator<TxRxParamsSequence>::Validator;

    void validate(const TxRxParamsSequence &txRxs) override {
        // Validation according to us4oem technote
        const auto decimationFactor = txRxs[0].getRxDecimationFactor();
        const auto startSample = txRxs[0].getRxSampleRange().start();
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            if(!op.isNOP()) {
                auto firingStr = ::arrus::format(" (firing {})", firing);

                // Tx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getTxAperture().size(), size_t(Us4OEMImpl::N_TX_CHANNELS),
                    firingStr);
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getTxDelays().size(), size_t(Us4OEMImpl::N_TX_CHANNELS),
                    firingStr);
                ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_VM(
                    op.getTxDelays(),
                    Us4OEMImpl::MIN_TX_DELAY, Us4OEMImpl::MAX_TX_DELAY,
                    firingStr);

                // Tx - pulse
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getTxPulse().getCenterFrequency(),
                    Us4OEMImpl::MIN_TX_FREQUENCY, Us4OEMImpl::MAX_TX_FREQUENCY,
                    firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getTxPulse().getNPeriods(), 0.0f, 32.0f, firingStr);
                float ignore = 0.0f;
                float fractional = std::modf(op.getTxPulse().getNPeriods(), &ignore);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    (fractional == 0.0f || fractional == 0.5f),
                    (firingStr + ", n periods"));

                // Rx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getRxAperture().size(), size_t(Us4OEMImpl::N_ADDR_CHANNELS), firingStr);
                size_t numberOfActiveRxChannels = std::accumulate(
                    std::begin(op.getRxAperture()), std::end(op.getRxAperture()), 0);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    numberOfActiveRxChannels, size_t(0), size_t(32), firingStr);
                uint32 numberOfSamples = op.getNumberOfSamples();
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                // should be enough for condition rxTime < 4000 [us]
                    numberOfSamples, Us4OEMImpl::MIN_NSAMPLES, Us4OEMImpl::MAX_NSAMPLES, firingStr);
                ARRUS_VALIDATOR_EXPECT_DIVISIBLE_M(
                    numberOfSamples, 64, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getRxDecimationFactor(), 0, 5, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getPri(),
                    Us4OEMImpl::MIN_PRI, Us4OEMImpl::MAX_PRI,
                    firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    op.getRxDecimationFactor() == decimationFactor,
                    "Decimation factor should be the same for all operations." + firingStr
                );
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    op.getRxSampleRange().start() == startSample,
                    "Start sample should be the same for all operations." + firingStr
                );
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    (op.getRxPadding() == ::arrus::Tuple<ChannelIdx>{0, 0}),
                    ("Rx padding is not allowed for us4oems. " + firingStr)
                );
            }
        }
    }
};

std::pair<float, float> getTgcMinMax(uint16 pgaGain, uint16 lnaGain) {
    float max = float(pgaGain) + float(lnaGain);
    return std::make_pair(max - 40, max);
}

class TGCCurveValidator : public Validator<::arrus::ops::us4r::TGCCurve> {
public:
    TGCCurveValidator(const std::string &componentName, uint16 pgaGain, uint16 lnaGain)
        : Validator(componentName), pgaGain(pgaGain), lnaGain(lnaGain) {}

    void validate(const ops::us4r::TGCCurve &tgcCurve) override {
        if(pgaGain != 30 || lnaGain != 24) {
            ARRUS_VALIDATOR_EXPECT_TRUE_M(
                tgcCurve.empty(),
                "Currently TGC is supported only for "
                "PGA gain 30 dB, LNA gain 24 dB");
        } else {
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(
                tgcCurve.size(), size_t(0), size_t(1022));
            auto[min, max] = getTgcMinMax(pgaGain, lnaGain);
            ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_V(tgcCurve, min, max);
        }
    }

private:
    uint16 pgaGain, lnaGain;
};

std::tuple<FrameChannelMapping::Handle, std::vector<std::vector<DataTransfer>>, float>
Us4OEMImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                            const ops::us4r::TGCCurve &tgc, uint16 nRepeats,
                            std::optional<float> fri) {
    // TODO initialize module: reset all parameters (turn off TGC, DTGC, ActiveTermination, etc.)
    // This probably should be implemented in IUs4OEMInitializer

    std::vector<std::vector<DataTransfer>> dataTransfers;

    // Validate input sequence and parameters.
    std::string deviceIdStr = getDeviceId().toString();
    Us4OEMTxRxValidator seqValidator(format("{} tx rx sequence", deviceIdStr));
    seqValidator.validate(seq);
    seqValidator.throwOnErrors();

    TGCCurveValidator tgcValidator(format("{} tgc samples", deviceIdStr), pgaGain, lnaGain);
    tgcValidator.validate(tgc);
    tgcValidator.throwOnErrors();

    // General sequence parameters.
    auto nOps = static_cast<uint16>(seq.size());
    ARRUS_REQUIRES_AT_MOST(nOps * nRepeats, 16384,
                           ::arrus::format(
                               "Exceeded the maximum ({}) number of firings: {}",
                               16384, nOps * nRepeats));

    ius4oem->SetNumberOfFirings(nOps * nRepeats);
    ius4oem->ClearScheduledReceive();

    auto[rxMappings, rxApertures, fcm] = setRxMappings(seq);

    // helper data
    const std::bitset<N_ADDR_CHANNELS> emptyAperture;
    const std::bitset<N_ACTIVE_CHANNEL_GROUPS> emptyChannelGroups;
    // us4oem rxdma output address
    size_t outputAddress = 0;

    uint16 transferFiringStart = 0;
    size_t transferAddressStart = 0;

    // Program Tx/rx sequence
    for(uint16 seqIdx = 0; seqIdx < nRepeats; ++seqIdx) {
        logger->log(LogSeverity::TRACE, format("Setting tx/rx sequence {} of {}", seqIdx + 1, nRepeats));
        for(uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
            uint16 firing = (uint16)(seqIdx * seq.size() + opIdx);
            auto const &op = seq[opIdx];
            bool checkpoint = op.isCheckpoint();
            if(op.isNOP()) {
                logger->log(LogSeverity::TRACE,
                            format("Setting tx/rx {}: NOP {}", opIdx, ::arrus::toString(op)));
            } else {
                logger->log(LogSeverity::DEBUG,
                            arrus::format("Setting tx/rx {}: {}", opIdx, ::arrus::toString(op)));
            }
            auto[startSample, endSample] = op.getRxSampleRange().asPair();
            size_t nSamples = endSample - startSample;
            float rxTime = getRxTime(nSamples);
            size_t nBytes = nSamples * N_RX_CHANNELS * sizeof(OutputDType);
            auto rxMapId = rxMappings.find(opIdx)->second;

            if(op.isNOP()) {
                ius4oem->SetActiveChannelGroup(emptyChannelGroups, firing);
                // Intentionally filtering empty aperture to reduce possibility of a mistake.
                auto txAperture = filterAperture(emptyAperture);
                auto rxAperture = filterAperture(emptyAperture);

                // Intentionally validating the apertures, to reduce possibility of mistake.
                validateAperture(txAperture);
                ius4oem->SetTxAperture(txAperture, firing);
                validateAperture(rxAperture);
                ius4oem->SetRxAperture(rxAperture, firing);
            } else {
                // active channel groups already remapped in constructor
                ius4oem->SetActiveChannelGroup(activeChannelGroups, firing);

                auto txAperture = filterAperture(::arrus::toBitset<N_TX_CHANNELS>(op.getTxAperture()));
                auto rxAperture = filterAperture(rxApertures[opIdx]);

                // Intentionally validating tx apertures, to reduce the risk of mistake channel activation
                // (e.g. the masked one).
                validateAperture(txAperture);
                ius4oem->SetTxAperture(txAperture, firing);
                validateAperture(rxAperture);
                ius4oem->SetRxAperture(rxAperture, firing);
            }

            // Delays
            uint8 txChannel = 0;
            for(bool bit : op.getTxAperture()) {
                float txDelay = 0;
                if(bit && !::arrus::setContains(this->channelsMask, txChannel)) {
                    txDelay = op.getTxDelays()[txChannel];
                }
                ius4oem->SetTxDelay(txChannel, txDelay, firing);
                ++txChannel;
            }
            ius4oem->SetTxFreqency(op.getTxPulse().getCenterFrequency(), firing);
            ius4oem->SetTxHalfPeriods(static_cast<uint8>(op.getTxPulse().getNPeriods() * 2), firing);
            ius4oem->SetTxInvert(op.getTxPulse().isInverse(), firing);
            ius4oem->SetRxTime(rxTime, firing);
            ius4oem->SetRxDelay(Us4OEMImpl::RX_DELAY, firing);
            setTGC(tgc, firing);

            ARRUS_REQUIRES_AT_MOST(outputAddress + nBytes, DDR_SIZE,
                                   ::arrus::format("Total data size cannot exceed 4GiB (device {})",
                                                   getDeviceId().toString()));

            std::optional<std::function<void()>> callback = std::nullopt;

            if(checkpoint && op.getCallback().has_value()) {
                callback = [this, seqIdx, op]() {
                    op.getCallback().value()(this->getDeviceId().getOrdinal(), seqIdx);
                };
            }
            if(op.isRxNOP() && !this->isMaster()) {
                // TODO reduce the size of data acquired for master  the rx nop to small number of samples
                // (e.g. 64)
                // TODO add optional configuration "is metadata"
                ius4oem->ScheduleReceive(firing, outputAddress, nSamples,
                                         SAMPLE_DELAY + startSample,
                                         op.getRxDecimationFactor() - 1,
                                         rxMapId, callback.value_or(nullptr));
            }
            else {
                // Also, allows rx nops for master module.
                // Master module gathers frame metadata, so we cannot miss any of them
                ius4oem->ScheduleReceive(firing, outputAddress, nSamples,
                                         SAMPLE_DELAY + startSample,
                                         op.getRxDecimationFactor() - 1,
                                         rxMapId, callback.value_or(nullptr));
                outputAddress += nBytes;
            }

            if(opIdx == nOps-1) {
                // The size of the chunk.
                auto size = outputAddress - transferAddressStart;
                // Where the chunk starts.
                auto srcAddress = transferAddressStart;
                // TODO replace "DataTransfer" With a "Variable" in the Us4OEM memory
                std::vector<DataTransfer> dataTransferSection = {
                    DataTransfer(
                        [=](uint8_t *dstAddress) {
                            if(size > 0) {
                                this->transferData(dstAddress, size, srcAddress);
                            }
                        },
                        size, srcAddress, firing)
                };
                dataTransfers.push_back(dataTransferSection);
                transferAddressStart = outputAddress;
                transferFiringStart = firing + 1;
            }
        }
    }
    ius4oem->EnableTransmit();

    // Set frame repetition interval if possible.
    float totalPri = 0.0f;
    for(auto &op : seq) {
        totalPri += op.getPri();
    }
    std::optional<float> lastPriExtend = std::nullopt;
    if(fri.has_value()) {
        if(totalPri < fri.value()) {
            lastPriExtend = totalPri - fri.value();
        } else {
            // TODO move this condition to sequence validator
            throw IllegalArgumentException(
                arrus::format("Frame repetition interval cannot be set, "
                              "sequence total pri is equal {}", totalPri));
        }
    }

    // Program triggers
    ius4oem->SetNTriggers(nOps * nRepeats);
    for(uint16 seqIdx = 0; seqIdx < nRepeats; ++seqIdx) {
        logger->log(LogSeverity::TRACE, format("Programming triggers {} of {}", seqIdx + 1, nRepeats));
        for(uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
            uint16 firing = (uint16) (seqIdx * seq.size() + opIdx);
            auto const &op = seq[opIdx];
            bool checkpoint = op.isCheckpoint();

            float pri = op.getPri();
            if(opIdx == nOps - 1 && lastPriExtend.has_value()) {
                pri += lastPriExtend.value();
            }
            ius4oem->SetTrigger(static_cast<short>(pri * 1e6), checkpoint, firing);
        }
    }
    ius4oem->EnableSequencer();
    return {std::move(fcm), std::move(dataTransfers), totalPri};
}

std::tuple<
    std::unordered_map<uint16, uint16>,
    std::vector<Us4OEMImpl::Us4rBitMask>,
    FrameChannelMapping::Handle>
Us4OEMImpl::setRxMappings(const std::vector<TxRxParameters> &seq) {
    // a map: op ordinal number -> rx map id
    std::unordered_map<uint16, uint16> result;
    std::unordered_map<std::vector<uint8>, uint16, ContainerHash<std::vector<uint8>>> rxMappings;

    // FC mapping
    auto numberOfOutputFrames = getNumberOfNoRxNOPs(seq);
    if(this->isMaster()) {
        // We transfer all master module frames due to possible metadata stored in the frame.
        numberOfOutputFrames = ARRUS_SAFE_CAST(seq.size(), ChannelIdx);
    }
    FrameChannelMappingBuilder fcmBuilder(numberOfOutputFrames, N_RX_CHANNELS);

    // Rx apertures after taking into account possible conflicts in Rx channel
    // mapping.
    std::vector<Us4rBitMask> outputRxApertures;

    uint16 rxMapId = 0;
    uint16 opId = 0;
    uint16 noRxNopId = 0;

    for(const auto &op: seq) {
        // Considering rx nops: rx channel mapping will be equal [0, 1,.. 31].
        // uint8 is required by us4r API.
        std::vector<uint8> mapping;
        std::unordered_set<uint8> channelsUsed;

        // Convert rx aperture + channel mapping -> rx channel mapping
        std::bitset<N_ADDR_CHANNELS> outputRxAperture;

        uint8 channel = 0;
        uint8 onChannel = 0;

        bool isRxNop = true;
        for(const auto isOn : op.getRxAperture()) {
            if(isOn) {
                isRxNop = false;
                ARRUS_REQUIRES_TRUE_E(
                    onChannel < N_RX_CHANNELS,
                    ArrusException("Up to 32 active rx channels can be set."));

                auto rxChannel = channelMapping[channel];
                rxChannel = rxChannel % N_RX_CHANNELS;
                if(!setContains(channelsUsed, rxChannel)
                   && !setContains(this->channelsMask, channel)) {
                    // STRATEGY: if there are conflicting/masked rx channels, keep the
                    // first one (with the lowest channel number), turn off all
                    // the rest.
                    // Turn off conflicting channels
                    outputRxAperture[channel] = true;
                }
                mapping.push_back(rxChannel);
                channelsUsed.insert(rxChannel);

                auto frameNumber = noRxNopId;
                if(this->isMaster()) {
                    frameNumber = opId;
                }
                fcmBuilder.setChannelMapping(frameNumber, onChannel, frameNumber, (int8) (mapping.size() - 1));

                ++onChannel;
            }
            ++channel;
        }
        outputRxApertures.push_back(outputRxAperture);

        // Move all the non active channels to the end of the mapping
        for(uint8 i = 0; i < N_RX_CHANNELS; ++i) {
            if(!setContains(channelsUsed, i)) {
                mapping.push_back(i);
            }
        }

        auto mappingIt = rxMappings.find(mapping);
        if(mappingIt == std::end(rxMappings)) {
            // Create new Rx channel mapping.
            rxMappings.emplace(mapping, rxMapId);
            result.emplace(opId, rxMapId);
            // Set channel mapping
            ARRUS_REQUIRES_TRUE(mapping.size() == N_RX_CHANNELS,
                                arrus::format(
                                    "Invalid size of the RX "
                                    "channel mapping to set: {}",
                                    mapping.size()));
            ARRUS_REQUIRES_TRUE(
                rxMapId < 128,
                arrus::format("128 different rx mappings can be loaded only"
                              ", deviceId: {}.", getDeviceId().toString()));
            ius4oem->SetRxChannelMapping(mapping, rxMapId);
            ++rxMapId;
        } else {
            // Use the existing one.
            result.emplace(opId, mappingIt->second);
        }
        ++opId;
        if(!isRxNop) {
            ++noRxNopId;
        }
    }
    return {result, outputRxApertures, fcmBuilder.build()};
}

double Us4OEMImpl::getSamplingFrequency() {
    return Us4OEMImpl::SAMPLING_FREQUENCY;
}

float Us4OEMImpl::getRxTime(size_t nSamples) {
    return nSamples / Us4OEMImpl::SAMPLING_FREQUENCY
           + Us4OEMImpl::RX_TIME_EPSILON;
}

void Us4OEMImpl::setTGC(const ops::us4r::TGCCurve &tgc, uint16 firing) {
    if(tgc.empty()) {
        ius4oem->TGCDisable();
    } else {
        ius4oem->TGCEnable();

        static const std::vector<float> tgcChar =
            {14.000f, 14.001f, 14.002f, 14.003f, 14.024f, 14.168f, 14.480f, 14.825f,
             15.234f, 15.770f, 16.508f, 17.382f, 18.469f, 19.796f, 20.933f, 21.862f,
             22.891f, 24.099f, 25.543f, 26.596f, 27.651f, 28.837f, 30.265f, 31.690f,
             32.843f, 34.045f, 35.543f, 37.184f, 38.460f, 39.680f, 41.083f, 42.740f,
             44.269f, 45.540f, 46.936f, 48.474f, 49.895f, 50.966f, 52.083f, 53.256f,
             54.0f};
        auto actualTGC = ::arrus::interpolate1d(
            tgcChar,
            ::arrus::getRange<float>(14, 55, 1.0),
            tgc);
        for(auto &val: actualTGC) {
            val = (val - 14.0f) / 40.0f;
        }
        ius4oem->TGCSetSamples(actualTGC, firing);
    }
}

std::bitset<Us4OEMImpl::N_ADDR_CHANNELS>
Us4OEMImpl::filterAperture(std::bitset<N_ADDR_CHANNELS> aperture) {
    for(auto channel : this->channelsMask) {
        aperture[channel] = false;
    }
    return aperture;
}

void
Us4OEMImpl::validateAperture(const std::bitset<N_ADDR_CHANNELS> &aperture) {
    for(auto channel : this->channelsMask) {
        if(aperture[channel]) {
            throw ArrusException(
                ::arrus::format("Attempted to set masked channel: {}", channel)
            );
        }
    }
}

void Us4OEMImpl::transferData(uint8_t *dstAddress, size_t size, size_t srcAddress) {
    // Maximum transfer part: 64 MB TODO (MB or MiB?)
    constexpr size_t MAX_TRANSFER_SIZE = 64*1000*1000;
    size_t transferredSize = 0;
    while(transferredSize < size) {
        size_t chunkSize = std::min(MAX_TRANSFER_SIZE, size - transferredSize);
        ius4oem->TransferRXBufferToHost(dstAddress, chunkSize, srcAddress);
        transferredSize += chunkSize;
    }
}

void Us4OEMImpl::start() {
    this->startTrigger();
}

void Us4OEMImpl::stop() {
    this->stopTrigger();
}

void Us4OEMImpl::syncTrigger() {
    this->ius4oem->TriggerSync();
}

void Us4OEMImpl::registerOutputBuffer(Us4ROutputBuffer *outputBuffer, const std::vector<std::vector<DataTransfer>> &transfers) {
    // Assuming here that each data transfer here will have exactly a single element.
    std::vector<DataTransfer> us4oemTransfers;
    for(auto &transfer: transfers) {
        us4oemTransfers.push_back(transfer[0]);
    }

    // Each transfer should have the same size.
    std::unordered_set<size_t> sizes;
    for(auto &transfer: us4oemTransfers){
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
    const auto rxBufferSize = ARRUS_SAFE_CAST(us4oemTransfers.size(), uint16);
    const uint16 hostBufferSize = outputBuffer->getNumberOfElements();
    const Ordinal ordinal = getDeviceId().getOrdinal();

    // Prepare host buffers
    uint16 hostElement = 0;
    uint16 rxElement = 0;
    while(hostElement < hostBufferSize) {
        auto dstAddress = outputBuffer->getAddress(hostElement, ordinal);
        auto srcAddress = us4oemTransfers[rxElement].getSrcAddress();
        logger->log(LogSeverity::DEBUG, ::arrus::format("Preparing host buffer to {} from {}, size {}",
                                                        (size_t)dstAddress, (size_t)srcAddress, elementSize));
        this->ius4oem->PrepareHostBuffer(dstAddress, elementSize, srcAddress);
        ++hostElement;
        rxElement = (rxElement+1) % rxBufferSize;
    }

    // prepare transfers
    uint16 transferIdx = 0;
    uint16 startFiring = 0;

    for(auto &transfer: us4oemTransfers) {
        auto dstAddress = outputBuffer->getAddress(transferIdx, ordinal);
        auto srcAddress = transfer.getSrcAddress();
        auto endFiring = transfer.getFiring();


        this->ius4oem->PrepareTransferRXBufferToHost(
            transferIdx, dstAddress, elementSize, srcAddress);

        this->ius4oem->ScheduleTransferRXBufferToHost(
            endFiring, transferIdx,
            [this, outputBuffer, ordinal, transferIdx, startFiring,
                    endFiring, srcAddress, elementSize,
                    rxBufferSize, hostBufferSize,
                    element = transferIdx] () mutable {
                auto dstAddress = outputBuffer->getAddress((uint16)element, ordinal);
                this->ius4oem->MarkEntriesAsReadyForReceive(startFiring, endFiring);
                logger->log(LogSeverity::DEBUG, ::arrus::format("Rx Released: {}, {}", startFiring, endFiring));

                // Prepare transfer for the next iteration.
                this->ius4oem->PrepareTransferRXBufferToHost(
                    transferIdx, dstAddress, elementSize, srcAddress);
                this->ius4oem->ScheduleTransferRXBufferToHost(endFiring, transferIdx, nullptr);

                bool cont = outputBuffer->signal(ordinal, element, 0); // Also a callback function can be used here.
                if(!cont) {
                    logger->log(LogSeverity::DEBUG, "Output buffer shut down.");
                    return;
                }
                cont = outputBuffer->waitForRelease(ordinal, element, 0);

                if(!cont) {
                    logger->log(LogSeverity::DEBUG, "Output buffer shut down");
                    return;
                }
                this->ius4oem->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
                logger->log(LogSeverity::DEBUG, ::arrus::format("Host Released: {}, {}", startFiring, endFiring));
                element = (element + rxBufferSize) % hostBufferSize;
            }
        );
        startFiring = endFiring + 1;
        ++transferIdx;
    }
    // Register overflow callbacks (mark output buffer as invalid)

    this->ius4oem->RegisterReceiveOverflowCallback([this, outputBuffer] () {
        this->logger->log(LogSeverity::ERROR, "Rx buffer overflow, stopping the device.");
        if(this->isMaster()) {
            this->stop();
        }
        outputBuffer->markAsInvalid();
    });

    this->ius4oem->RegisterTransferOverflowCallback([this, outputBuffer] () {
        this->logger->log(LogSeverity::ERROR, "Host buffer overflow, stopping the device.");
        if(this->isMaster()) {
            this->stop();
        }
        outputBuffer->markAsInvalid();
    });
}

}
