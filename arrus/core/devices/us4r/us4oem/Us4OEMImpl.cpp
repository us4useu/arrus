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
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
                       const BitMask &activeChannelGroups,
                       std::vector<uint8_t> channelMapping,
                       uint16 pgaGain, uint16 lnaGain,
                       std::unordered_set<uint8_t> channelsMask,
                       Us4OEMSettings::ReprogrammingMode reprogrammingMode)
    : Us4OEMImplBase(id), logger{getLoggerFactory()->getLogger()},
      ius4oem(std::move(ius4oem)),
      channelMapping(std::move(channelMapping)),
      channelsMask(std::move(channelsMask)),
      pgaGain(pgaGain), lnaGain(lnaGain),
      reprogrammingMode(reprogrammingMode) {

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
            "No channel masking will be applied for {}",
            ::arrus::toString(id)));
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
        std::cerr
            << arrus::format("Exception while calling us4oem destructor: {}",
                             e.what())
            << std::endl;
    }
    logger->log(LogSeverity::DEBUG, arrus::format("Us4OEM handle destroyed."));
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
                    op.getTxAperture().size(),
                    size_t(Us4OEMImpl::N_TX_CHANNELS),
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
                float fractional = std::modf(op.getTxPulse().getNPeriods(),
                                             &ignore);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    (fractional == 0.0f || fractional == 0.5f),
                    (firingStr + ", n periods"));

                // Rx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getRxAperture().size(),
                    size_t(Us4OEMImpl::N_ADDR_CHANNELS), firingStr);
                size_t numberOfActiveRxChannels = std::accumulate(
                    std::begin(op.getRxAperture()),
                    std::end(op.getRxAperture()), 0);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    numberOfActiveRxChannels, size_t(0), size_t(32), firingStr);
                uint32 numberOfSamples = op.getNumberOfSamples();
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                // should be enough for condition rxTime < 4000 [us]
                    numberOfSamples, Us4OEMImpl::MIN_NSAMPLES,
                    Us4OEMImpl::MAX_NSAMPLES, firingStr);
                ARRUS_VALIDATOR_EXPECT_DIVISIBLE_M(
                    numberOfSamples, 64u, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getRxDecimationFactor(), 0, 5, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getPri(),
                    Us4OEMImpl::MIN_PRI, Us4OEMImpl::MAX_PRI,
                    firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    op.getRxDecimationFactor() == decimationFactor,
                    "Decimation factor should be the same for all operations." +
                    firingStr
                );
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    op.getRxSampleRange().start() == startSample,
                    "Start sample should be the same for all operations." +
                    firingStr
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
    TGCCurveValidator(const std::string &componentName, uint16 pgaGain,
                      uint16 lnaGain)
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

std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
Us4OEMImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                            const ops::us4r::TGCCurve &tgc, uint16 rxBufferSize,
                            uint16 batchSize, std::optional<float> sri,
                            bool triggerSync) {
    // TODO initialize module: reset all parameters (turn off TGC, DTGC, ActiveTermination, etc.)
    // Validate input sequence and parameters.
    std::string deviceIdStr = getDeviceId().toString();
    Us4OEMTxRxValidator seqValidator(format("{} tx rx sequence", deviceIdStr));
    seqValidator.validate(seq);
    seqValidator.throwOnErrors();

    TGCCurveValidator tgcValidator(format("{} tgc samples", deviceIdStr),
                                   pgaGain, lnaGain);
    tgcValidator.validate(tgc);
    tgcValidator.throwOnErrors();

    // General sequence parameters.
    auto nOps = static_cast<uint16>(seq.size());
    ARRUS_REQUIRES_AT_MOST(nOps * batchSize, 1024, ::arrus::format(
        "Exceeded the maximum ({}) number of firings: {}", 1024, nOps));
    ARRUS_REQUIRES_AT_MOST(nOps * batchSize * rxBufferSize, 16384,
                           ::arrus::format(
                               "Exceeded the maximum ({}) number of triggers: {}",
                               16384, nOps * batchSize * rxBufferSize));

    ius4oem->SetNumberOfFirings(nOps * batchSize);
    ius4oem->ClearScheduledReceive();
    ius4oem->ResetCallbacks();
    auto[rxMappings, rxApertures, fcm] = setRxMappings(seq);

    // helper data
    const std::bitset<N_ADDR_CHANNELS> emptyAperture;
    const std::bitset<N_ACTIVE_CHANNEL_GROUPS> emptyChannelGroups;

    // Program Tx/rx sequence ("firings")
    setTGC(tgc);
    for(uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
        logger->log(LogSeverity::TRACE, format("Setting tx/rx: {}", opIdx));
        auto const &op = seq[opIdx];
        if(op.isNOP()) {
            logger->log(LogSeverity::TRACE,
                        format("Setting tx/rx {}: NOP {}", opIdx,
                               ::arrus::toString(op)));
        } else {
            logger->log(LogSeverity::DEBUG,
                        arrus::format("Setting tx/rx {}: {}", opIdx,
                                      ::arrus::toString(op)));
        }
        auto sampleRange = op.getRxSampleRange().asPair();
        auto endSample = std::get<1>(sampleRange);
        float rxTime = getRxTime(endSample, op.getRxDecimationFactor());

        // Computing total TX/RX time
        float txrxTime = 0.0f;
        txrxTime = getTxRxTime(rxTime);
        // receive time + reprogramming time
        if(txrxTime > op.getPri()) {
            throw IllegalArgumentException(
                ::arrus::format(
                        "Total time required for a single TX/RX ({}) should not exceed PRI ({})",
                        txrxTime, op.getPri()));
        }
        if(op.isNOP()) {
            ius4oem->SetActiveChannelGroup(emptyChannelGroups, opIdx);
            // Intentionally filtering empty aperture to reduce possibility of a mistake.
            auto txAperture = filterAperture(emptyAperture);
            auto rxAperture = filterAperture(emptyAperture);

            // Intentionally validating the apertures, to reduce possibility of mistake.
            validateAperture(txAperture);
            ius4oem->SetTxAperture(txAperture, opIdx);
            validateAperture(rxAperture);
            ius4oem->SetRxAperture(rxAperture, opIdx);
        } else {
            // active channel groups already remapped in constructor
            ius4oem->SetActiveChannelGroup(activeChannelGroups, opIdx);

            auto txAperture = filterAperture(
                ::arrus::toBitset<N_TX_CHANNELS>(op.getTxAperture()));
            auto rxAperture = filterAperture(rxApertures[opIdx]);

            // Intentionally validating tx apertures, to reduce the risk of mistake channel activation
            // (e.g. the masked one).
            validateAperture(txAperture);
            ius4oem->SetTxAperture(txAperture, opIdx);
            validateAperture(rxAperture);
            ius4oem->SetRxAperture(rxAperture, opIdx);
        }

        // Delays
        uint8 txChannel = 0;
        for(bool bit : op.getTxAperture()) {
            float txDelay = 0;
            if(bit && !::arrus::setContains(this->channelsMask, txChannel)) {
                txDelay = op.getTxDelays()[txChannel];
            }
            ius4oem->SetTxDelay(txChannel, txDelay, opIdx);
            ++txChannel;
        }
        ius4oem->SetTxFreqency(op.getTxPulse().getCenterFrequency(), opIdx);
        ius4oem->SetTxHalfPeriods(
            static_cast<uint8>(op.getTxPulse().getNPeriods() * 2), opIdx);
        ius4oem->SetTxInvert(op.getTxPulse().isInverse(), opIdx);
        ius4oem->SetRxTime(rxTime, opIdx);
        ius4oem->SetRxDelay(Us4OEMImpl::RX_DELAY, opIdx);
    }

    // Program data acquisitions ("ScheduleReceive" part)
    // element == the result data frame of the given operations sequence
    // Buffer elements.
    // The below code fills the us4oem memory with the acquired data.
    // us4oem rxdma output address
    size_t outputAddress = 0;
    size_t transferAddressStart = 0;

    uint16 firing = 0;
    std::vector<Us4OEMBufferElement> rxBufferElements;
    for(uint16 batchIdx = 0; batchIdx < rxBufferSize; ++batchIdx) {
        // Total number of samples in a single batch.
        unsigned int totalNSamples = 0;
        // Batch elements.
        for(uint16 batchElementIdx = 0;
            batchElementIdx < batchSize; ++batchElementIdx) {
            // Element operation.
            for(uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
                firing = opIdx + (batchElementIdx * nOps) +
                         (batchIdx * nOps * batchSize);
                auto const &op = seq[opIdx];
                auto[startSample, endSample] = op.getRxSampleRange().asPair();
                size_t nSamples = endSample - startSample;
                size_t nBytes = nSamples * N_RX_CHANNELS * sizeof(OutputDType);
                auto rxMapId = rxMappings.find(opIdx)->second;

                ARRUS_REQUIRES_AT_MOST(
                    outputAddress + nBytes, DDR_SIZE,
                    ::arrus::format(
                        "Total data size cannot exceed 4GiB (device {})",
                        getDeviceId().toString()));

                if(op.isRxNOP() && !this->isMaster()) {
                    // TODO reduce the size of data acquired from master rx nops to small number of samples
                    // (e.g. 64)
                    ius4oem->ScheduleReceive(firing, outputAddress, nSamples,
                                             SAMPLE_DELAY + startSample,
                                             op.getRxDecimationFactor() - 1,
                                             rxMapId, nullptr);
                } else {
                    // Also, allows rx nops for master module.
                    // Master module gathers frame metadata, so we cannot miss any of them
                    ius4oem->ScheduleReceive(firing, outputAddress, nSamples,
                                             SAMPLE_DELAY + startSample,
                                             op.getRxDecimationFactor() - 1,
                                             rxMapId, nullptr);
                    outputAddress += nBytes;
                    totalNSamples += (unsigned) nSamples;
                }
            }
        }
        // The size of the chunk, in number of BYTES.
        auto size = outputAddress - transferAddressStart;
        // Where the chunk starts.
        auto srcAddress = transferAddressStart;
        transferAddressStart = outputAddress;
        framework::NdArray::Shape shape{totalNSamples, N_RX_CHANNELS};
        rxBufferElements.emplace_back(srcAddress, size, firing, shape,
                                      NdArrayDataType);
    }
    ius4oem->EnableTransmit();

    // Set frame repetition interval if possible.
    float totalPri = 0.0f;
    for(auto &op : seq) {
        totalPri += op.getPri();
    }
    std::optional<float> lastPriExtend = std::nullopt;
    if(sri.has_value()) {
        if(totalPri < sri.value()) {
            lastPriExtend = sri.value() - totalPri;
        } else {
            // TODO move this condition to sequence validator
            throw IllegalArgumentException(
                arrus::format("Sequence repetition interval {} cannot be set, "
                              "sequence total pri is equal {}",
                              sri.value(), totalPri));
        }
    }

    // Program triggers
    ius4oem->SetNTriggers(nOps * batchSize * rxBufferSize);
    firing = 0;
    for(uint16 batchIdx = 0; batchIdx < rxBufferSize; ++batchIdx) {
        for(uint16 batchElementIdx = 0;
            batchElementIdx < batchSize; ++batchElementIdx) {
            for(uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
                firing = (uint16) (opIdx + batchElementIdx * nOps +
                                   batchIdx * nOps * batchSize);
                auto const &op = seq[opIdx];
                // checkpoint only when it is the last operation of the last batch element
                bool checkpoint = triggerSync && (opIdx == seq.size() - 1 &&
                                                  batchElementIdx ==
                                                  batchSize - 1);
                float pri = op.getPri();
                if(opIdx == nOps - 1 && lastPriExtend.has_value()) {
                    pri += lastPriExtend.value();
                }
                auto priMs = static_cast<unsigned int>(std::round(pri * 1e6));
                ius4oem->SetTrigger(priMs, checkpoint, firing);
            }
        }
    }
    return {Us4OEMBuffer(rxBufferElements), std::move(fcm)};
}

float Us4OEMImpl::getTxRxTime(float rxTime) const {
    float txrxTime = 0.0f;
    if(reprogrammingMode == Us4OEMSettings::ReprogrammingMode::SEQUENTIAL) {
        txrxTime = rxTime + SEQUENCER_REPROGRAMMING_TIME;
    }
    else if(reprogrammingMode == Us4OEMSettings::ReprogrammingMode::PARALLEL) {
        txrxTime = std::max(rxTime, SEQUENCER_REPROGRAMMING_TIME);
    }
    else {
        throw IllegalArgumentException(
            ::arrus::format(
                    "Unrecognized reprogramming mode: {}",
                    static_cast<size_t>(reprogrammingMode))
        );
    }
    return txrxTime;
}

std::tuple<
    std::unordered_map<uint16, uint16>,
    std::vector<Us4OEMImpl::Us4OEMBitMask>,
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
    std::vector<Us4OEMBitMask> outputRxApertures;

    uint16 rxMapId = 0;
    uint16 opId = 0;
    uint16 noRxNopId = 0;

    for(const auto &op: seq) {
        // Considering rx nops: rx channel mapping will be equal [0, 1,.. 31].

        // Index of rx aperture channel (0, 1...32) -> us4oem physical channel
        // nullopt means that given channel is missing (conflicting with some other channel or is masked)
        std::vector<std::optional<uint8>> mapping;
        std::unordered_set<uint8> channelsUsed;

        // Convert rx aperture + channel mapping -> new rx aperture (with conflicting channels turned off).
        std::bitset<N_ADDR_CHANNELS> outputRxAperture;

        // Us4OEM channel number: values from 0-127
        uint8 channel = 0;
        // Number of Us4OEM active channel, values from 0-31
        uint8 onChannel = 0;

        bool isRxNop = true;
        for(const auto isOn : op.getRxAperture()) {
            if(isOn) {
                isRxNop = false;
                ARRUS_REQUIRES_TRUE_E(
                    onChannel < N_RX_CHANNELS,
                    ArrusException("Up to 32 active rx channels can be set."));

                // Physical channel number, values 0-31
                auto rxChannel = channelMapping[channel];
                rxChannel = rxChannel % N_RX_CHANNELS;
                if(!setContains(channelsUsed, rxChannel) &&
                   !setContains(this->channelsMask, channel)) {
                    // This channel is OK.
                    // STRATEGY: if there are conflicting/masked rx channels, keep the
                    // first one (with the lowest channel number), turn off all
                    // the rest. Turn off conflicting channels.
                    outputRxAperture[channel] = true;
                    mapping.emplace_back(rxChannel);
                    channelsUsed.insert(rxChannel);
                } else {
                    // This channel is not OK.
                    mapping.emplace_back(std::nullopt);
                }
                auto frameNumber = noRxNopId;
                if(this->isMaster()) {
                    frameNumber = opId;
                }
                fcmBuilder.setChannelMapping(frameNumber, onChannel,
                                             frameNumber,
                                             (int8) (mapping.size() - 1));
                ++onChannel;
            }
            ++channel;
        }
        outputRxApertures.push_back(outputRxAperture);

        // Replace invalid channels with unused channels
        std::list<uint8> unusedChannels;
        for(uint8 i = 0; i < N_RX_CHANNELS; ++i) {
            if(!setContains(channelsUsed, i)) {
                unusedChannels.push_back(i);
            }
        }
        std::vector<uint8> rxMapping;
        for(auto &dstChannel: mapping) {
            if(!dstChannel.has_value()) {
                rxMapping.push_back(unusedChannels.front());
                unusedChannels.pop_front();
            } else {
                rxMapping.push_back(dstChannel.value());
            }
        }
        // Move all the non-active channels to the end of mapping.
        while(rxMapping.size() != 32) {
            rxMapping.push_back(unusedChannels.front());
            unusedChannels.pop_front();
        }

        auto mappingIt = rxMappings.find(rxMapping);
        if(mappingIt == std::end(rxMappings)) {
            // Create new Rx channel mapping.
            rxMappings.emplace(rxMapping, rxMapId);
            result.emplace(opId, rxMapId);
            // Set channel mapping
            ARRUS_REQUIRES_TRUE(rxMapping.size() == N_RX_CHANNELS,
                                arrus::format(
                                    "Invalid size of the RX "
                                    "channel mapping to set: {}",
                                    rxMapping.size()));
            ARRUS_REQUIRES_TRUE(
                rxMapId < 128,
                arrus::format("128 different rx mappings can be loaded only"
                              ", deviceId: {}.", getDeviceId().toString()));
            ius4oem->SetRxChannelMapping(rxMapping, rxMapId);
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

float Us4OEMImpl::getRxTime(size_t nSamples, uint32 decimationFactor) {
    return nSamples / (Us4OEMImpl::SAMPLING_FREQUENCY / decimationFactor) + RX_TIME_EPSILON;
}

void Us4OEMImpl::setTGC(const ops::us4r::TGCCurve &tgc) {
    if(tgc.empty()) {
        ius4oem->TGCDisable();
    } else {
        ius4oem->TGCEnable();

        static const std::vector<float> tgcChar =
            {14.000f, 14.001f, 14.002f, 14.003f, 14.024f, 14.168f, 14.480f,
             14.825f,
             15.234f, 15.770f, 16.508f, 17.382f, 18.469f, 19.796f, 20.933f,
             21.862f,
             22.891f, 24.099f, 25.543f, 26.596f, 27.651f, 28.837f, 30.265f,
             31.690f,
             32.843f, 34.045f, 35.543f, 37.184f, 38.460f, 39.680f, 41.083f,
             42.740f,
             44.269f, 45.540f, 46.936f, 48.474f, 49.895f, 50.966f, 52.083f,
             53.256f,
             54.0f};
        auto actualTGC = ::arrus::interpolate1d(
            tgcChar,
            ::arrus::getRange<float>(14, 55, 1.0),
            tgc);
        for(auto &val: actualTGC) {
            val = (val - 14.0f) / 40.0f;
        }
        // Currently setting firing parameter has no impact on the result
        // because TGC can be set only once for the whole sequence.
        ius4oem->TGCSetSamples(actualTGC, 0);
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

void Us4OEMImpl::start() {
    this->startTrigger();
}

void Us4OEMImpl::stop() {
    this->stopTrigger();
}

void Us4OEMImpl::syncTrigger() {
    this->ius4oem->TriggerSync();
}

void Us4OEMImpl::setTgcCurve(const ops::us4r::TGCCurve &tgc) {
    // Currently firing parameter doesn't matter.
    this->setTGC(tgc);
}

Ius4OEMRawHandle Us4OEMImpl::getIUs4oem() {
    return ius4oem.get();
}

void Us4OEMImpl::enableSequencer() {
    if(reprogrammingMode != Us4OEMSettings::ReprogrammingMode::SEQUENTIAL) {
        throw std::runtime_error("For us4R-api version 0.5.x only SEQUENTIAL reprogramming mode is available");
    }
    this->ius4oem->EnableSequencer();
}

std::vector<uint8_t> Us4OEMImpl::getChannelMapping() {
    return channelMapping;
}

}
