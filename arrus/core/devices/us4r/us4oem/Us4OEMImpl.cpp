#include "Us4OEMImpl.h"

#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/common/utils.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/common/interpolate.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/core/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/external/ius4oem/ActiveTerminationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem, const BitMask &activeChannelGroups,
                       std::vector<uint8_t> channelMapping, RxSettings rxSettings,
                       std::unordered_set<uint8_t> channelsMask, Us4OEMSettings::ReprogrammingMode reprogrammingMode,
                       bool externalTrigger = false)
    : Us4OEMImplBase(id), logger{getLoggerFactory()->getLogger()}, ius4oem(std::move(ius4oem)),
      channelMapping(std::move(channelMapping)), channelsMask(std::move(channelsMask)),
      reprogrammingMode(reprogrammingMode), rxSettings(std::move(rxSettings)), externalTrigger(externalTrigger) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());

    // This class stores reordered active groups of channels,
    // as presented in the IUs4OEM docs.
    static const std::vector<ChannelIdx> acgRemap = {0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15};
    auto acg = ::arrus::permute(activeChannelGroups, acgRemap);
    ARRUS_REQUIRES_TRUE(acg.size() == activeChannelGroups.size(),
                        arrus::format("Invalid number of active channels mask elements; the input has {}, expected: {}",
                                      acg.size(), activeChannelGroups.size()));
    this->activeChannelGroups = ::arrus::toBitset<N_ACTIVE_CHANNEL_GROUPS>(acg);

    if (this->channelsMask.empty()) {
        this->logger->log(LogSeverity::INFO,
                          ::arrus::format("No channel masking will be applied for {}", ::arrus::toString(id)));
    } else {
        this->logger->log(
            LogSeverity::INFO,
            ::arrus::format("Following us4oem channels will be turned off: {}", ::arrus::toString(this->channelsMask)));
    }
    setTestPattern(RxTestPattern::OFF);
    disableAfeDemod();
    setRxSettingsPrivate(this->rxSettings, true);
}

Us4OEMImpl::~Us4OEMImpl() {
    try {
        logger->log(LogSeverity::DEBUG, arrus::format("Destroying handle"));
    } catch (const std::exception &e) {
        std::cerr << arrus::format("Exception while calling us4oem destructor: {}", e.what()) << std::endl;
    }
    logger->log(LogSeverity::DEBUG, arrus::format("Us4OEM handle destroyed."));
}

bool Us4OEMImpl::isMaster() { return getDeviceId().getOrdinal() == 0; }

void Us4OEMImpl::startTrigger() {
    if (isMaster()) {
        ius4oem->TriggerStart();
    }
}

void Us4OEMImpl::stopTrigger() {
    if (isMaster()) {
        ius4oem->TriggerStop();
    }
}

uint16_t Us4OEMImpl::getAfe(uint8_t address) { return ius4oem->AfeReadRegister(0, address); }

void Us4OEMImpl::setAfe(uint8_t address, uint16_t value) {
    ius4oem->AfeWriteRegister(0, address, value);
    ius4oem->AfeWriteRegister(1, address, value);
}

void Us4OEMImpl::enableAfeDemod() { ius4oem->AfeDemodEnable(); }

void Us4OEMImpl::setAfeDemodConfig(uint8_t decInt, uint8_t decQuarters, const float *firCoeffs, uint16_t firLength,
                                   float freq) {
    ius4oem->AfeDemodConfig(decInt, decQuarters, firCoeffs, firLength, freq);
}

void Us4OEMImpl::setAfeDemodDefault() { ius4oem->AfeDemodSetDefault(); }

void Us4OEMImpl::setAfeDemodDecimationFactor(uint8_t integer) { ius4oem->AfeDemodSetDecimationFactor(integer); }

void Us4OEMImpl::setAfeDemodDecimationFactor(uint8_t integer, uint8_t quarters) {
    ius4oem->AfeDemodSetDecimationFactorQuarters(integer, quarters);
}

void Us4OEMImpl::setAfeDemodFrequency(float frequency) {
    // Note: us4r-api expects frequency in Hz.
    ius4oem->AfeDemodSetDemodFrequency(frequency / 1e6f);
    ius4oem->AfeDemodFsweepDisable();
}

void Us4OEMImpl::setAfeDemodFrequency(float startFrequency, float stopFrequency) {
    // Note: us4r-api expects frequency in Hz.
    ius4oem->AfeDemodSetDemodFrequency(startFrequency / 1e6f, stopFrequency / 1e6f);
    ius4oem->AfeDemodFsweepEnable();
}

float Us4OEMImpl::getAfeDemodStartFrequency() { return ius4oem->AfeDemodGetStartFrequency(); }

float Us4OEMImpl::getAfeDemodStopFrequency() { return ius4oem->AfeDemodGetStopFrequency(); }

void Us4OEMImpl::setAfeDemodFsweepROI(uint16_t startSample, uint16_t stopSample) {
    ius4oem->AfeDemodSetFsweepROI(startSample, stopSample);
}

void Us4OEMImpl::writeAfeFIRCoeffs(const int16_t *coeffs, uint16_t length) {
    ius4oem->AfeDemodWriteFirCoeffs(coeffs, length);
}

void Us4OEMImpl::writeAfeFIRCoeffs(const float *coeffs, uint16_t length) {
    ius4oem->AfeDemodWriteFirCoeffs(coeffs, length);
}

void Us4OEMImpl::setHpfCornerFrequency(uint32_t frequency) {
    uint8_t coefficient = 10;
    switch (frequency) {
    case 4520'000: coefficient = 2; break;
    case 2420'000: coefficient = 3; break;
    case 1200'000: coefficient = 4; break;
    case 600'000: coefficient = 5; break;
    case 300'000: coefficient = 6; break;
    case 180'000: coefficient = 7; break;
    case 80'000: coefficient = 8; break;
    case 40'000: coefficient = 9; break;
    case 20'000: coefficient = 10; break;
    default:
        throw ::arrus::IllegalArgumentException(::arrus::format("Unsupported HPF corner frequency: {}", frequency));
    }
    ius4oem->AfeEnableHPF();
    ius4oem->AfeSetHPFCornerFrequency(coefficient);
}

void Us4OEMImpl::disableHpf() { ius4oem->AfeDisableHPF(); }

void Us4OEMImpl::resetAfe() { ius4oem->AfeSoftReset(); }

class Us4OEMTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    using Validator<TxRxParamsSequence>::Validator;

    void validate(const TxRxParamsSequence &txRxs) {
        // Validation according to us4oem technote
        const auto decimationFactor = txRxs[0].getRxDecimationFactor();
        const auto startSample = txRxs[0].getRxSampleRange().start();
        for (size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            if (!op.isNOP()) {
                auto firingStr = ::arrus::format(" (firing {})", firing);

                // Tx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxAperture().size(), size_t(Us4OEMImpl::N_TX_CHANNELS), firingStr);
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxDelays().size(), size_t(Us4OEMImpl::N_TX_CHANNELS), firingStr);
                ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_VM(op.getTxDelays(), Us4OEMImpl::MIN_TX_DELAY,
                                                       Us4OEMImpl::MAX_TX_DELAY, firingStr);

                // Tx - pulse
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getTxPulse().getCenterFrequency(), Us4OEMImpl::MIN_TX_FREQUENCY,
                                                  Us4OEMImpl::MAX_TX_FREQUENCY, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getTxPulse().getNPeriods(), 0.0f, 32.0f, firingStr);
                float ignore = 0.0f;
                float fractional = std::modf(op.getTxPulse().getNPeriods(), &ignore);
                ARRUS_VALIDATOR_EXPECT_TRUE_M((fractional == 0.0f || fractional == 0.5f), (firingStr + ", n periods"));

                // Rx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getRxAperture().size(), size_t(Us4OEMImpl::N_ADDR_CHANNELS),
                                               firingStr);
                size_t numberOfActiveRxChannels =
                    std::accumulate(std::begin(op.getRxAperture()), std::end(op.getRxAperture()), 0);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(numberOfActiveRxChannels, size_t(0), size_t(32), firingStr);
                uint32 numberOfSamples = op.getNumberOfSamples();
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    // should be enough for condition rxTime < 4000 [us]
                    numberOfSamples, Us4OEMImpl::MIN_NSAMPLES, Us4OEMImpl::MAX_NSAMPLES, firingStr);
                ARRUS_VALIDATOR_EXPECT_DIVISIBLE_M(numberOfSamples, 64u, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getRxDecimationFactor(), 0, 10, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getPri(), Us4OEMImpl::MIN_PRI, Us4OEMImpl::MAX_PRI, firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getRxDecimationFactor() == decimationFactor,
                                              "Decimation factor should be the same for all operations." + firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getRxSampleRange().start() == startSample,
                                              "Start sample should be the same for all operations." + firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M((op.getRxPadding() == ::arrus::Tuple<ChannelIdx>{0, 0}),
                                              ("Rx padding is not allowed for us4oems. " + firingStr));
            }
        }
    }
};

std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
Us4OEMImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgc, uint16 rxBufferSize,
                            uint16 batchSize, std::optional<float> sri, bool triggerSync,
                            const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc) {
    std::unique_lock<std::mutex> lock{stateMutex};
    // Validate input sequence and parameters.
    std::string deviceIdStr = getDeviceId().toString();
    bool isDDCOn = ddc.has_value();
    Us4OEMTxRxValidator seqValidator(format("{} tx rx sequence", deviceIdStr));
    seqValidator.validate(seq);
    seqValidator.throwOnErrors();

    // General sequence parameters.
    auto nOps = static_cast<uint16>(seq.size());
    ARRUS_REQUIRES_AT_MOST(nOps, 1024, ::arrus::format("Exceeded the maximum ({}) number of firings: {}", 1024, nOps));
    ARRUS_REQUIRES_AT_MOST(
        nOps * batchSize * rxBufferSize, 16384,
        ::arrus::format("Exceeded the maximum ({}) number of triggers: {}", 16384, nOps * batchSize * rxBufferSize));

    RxSettingsBuilder rxSettingsBuilder(this->rxSettings);
    this->rxSettings = RxSettingsBuilder(this->rxSettings).setTgcSamples(tgc)->build();

    setTgcCurve(this->rxSettings);
    ius4oem->SetNumberOfFirings(nOps);
    ius4oem->ClearScheduledReceive();
    ius4oem->ResetCallbacks();
    auto [rxMappings, rxApertures, fcm] = setRxMappings(seq);
    // helper data
    const std::bitset<N_ADDR_CHANNELS> emptyAperture;
    const std::bitset<N_ACTIVE_CHANNEL_GROUPS> emptyChannelGroups;

    // Program Tx/rx sequence ("firings")
    for (uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
        logger->log(LogSeverity::TRACE, format("Setting tx/rx: {}", opIdx));
        auto const &op = seq[opIdx];
        if (op.isNOP()) {
            logger->log(LogSeverity::TRACE, format("Setting tx/rx {}: NOP {}", opIdx, ::arrus::toString(op)));
        } else {
            logger->log(LogSeverity::DEBUG, arrus::format("Setting tx/rx {}: {}", opIdx, ::arrus::toString(op)));
        }
        auto sampleRange = op.getRxSampleRange().asPair();
        auto endSample = std::get<1>(sampleRange);
        float decimationFactor = isDDCOn ? ddc->getDecimationFactor() : (float) op.getRxDecimationFactor();
        this->currentSamplingFrequency = SAMPLING_FREQUENCY / decimationFactor;
        float rxTime = getRxTime(endSample, this->currentSamplingFrequency);

        // Computing total TX/RX time
        float txrxTime = 0.0f;
        txrxTime = getTxRxTime(rxTime);
        // receive time + reprogramming time
        if (txrxTime > op.getPri()) {
            throw IllegalArgumentException(::arrus::format(
                "Total time required for a single TX/RX ({}) should not exceed PRI ({})", txrxTime, op.getPri()));
        }
        if (op.isNOP()) {
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
            auto txAperture = filterAperture(::arrus::toBitset<N_TX_CHANNELS>(op.getTxAperture()));
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
        for (bool bit : op.getTxAperture()) {
            float txDelay = 0;
            if (bit && !::arrus::setContains(this->channelsMask, txChannel)) {
                txDelay = op.getTxDelays()[txChannel];
            }
            ius4oem->SetTxDelay(txChannel, txDelay, opIdx);
            ++txChannel;
        }
        ius4oem->SetTxFreqency(op.getTxPulse().getCenterFrequency(), opIdx);
        ius4oem->SetTxHalfPeriods(static_cast<uint8>(op.getTxPulse().getNPeriods() * 2), opIdx);
        ius4oem->SetTxInvert(op.getTxPulse().isInverse(), opIdx);
        ius4oem->SetRxTime(rxTime, opIdx);
        ius4oem->SetRxDelay(RX_DELAY, opIdx);
    }

    // Program data acquisitions ("ScheduleReceive" part).
    // element == the result data frame of the given operations sequence
    // Buffer elements.
    // The below code programs us4OEM sequencer to fill the us4OEM memory with the acquired data.
    // us4oem RXDMA output address
    size_t outputAddress = 0;
    size_t transferAddressStart = 0;
    uint16 firing = 0;
    std::vector<Us4OEMBufferElement> rxBufferElements;
    // Assumption: all elements consists of the same parts.
    std::vector<Us4OEMBufferElementPart> rxBufferElementParts;

    for (uint16 batchIdx = 0; batchIdx < rxBufferSize; ++batchIdx) {
        // Total number of samples in a single batch.
        unsigned int totalNSamples = 0;
        // Sequences.
        for (uint16 seqIdx = 0; seqIdx < batchSize; ++seqIdx) {
            // Ops.
            for (uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
                firing = opIdx + seqIdx * nOps + batchIdx * batchSize * nOps;
                auto const &op = seq[opIdx];
                auto [startSample, endSample] = op.getRxSampleRange().asPair();
                size_t nSamples = endSample - startSample;
                auto rxMapId = rxMappings.find(opIdx)->second;

                // Start sample, after transforming to the system number of cycles.
                // The start sample should be provided to the us4r-api
                // as for the nominal sampling frequency of us4OEM, i.e. 65 MHz.
                // The ARRUS API assumes that the start sample and end sample are for the same
                // sampling frequency.
                uint32_t startSampleRaw = 0;
                // RX offset to the moment tx delay = 0.
                uint32_t sampleOffset = 0;
                // Number of samples to acquire to be set in us4r::IUS4OEM object.
                size_t nSamplesRaw = 0;
                // Number of bytes a single sample takes (e.g. RF: a single int16, IQ: a pair of int16)
                size_t sampleSize = 0;

                // Determine number of samples and offsets depending on whether hardware
                // DDC is on or off.
                if (isDDCOn) {
                    float decInt = 0;
                    float decFloat = modf(ddc->getDecimationFactor(), &decInt);
                    uint32_t div = 1;

                    if (decFloat == 0.5f) {
                        div = 2;
                    } else if (decFloat == 0.25f || decFloat == 0.75f) {
                        div = 4;
                    }

                    if (startSample != (startSample / div) * div) {
                        startSample = (startSample / div) * div;
                        this->logger->log(LogSeverity::WARNING,
                                          ::arrus::format("Decimation factor {} requires start offset to be multiple "
                                                          "of {}. Offset adjusted to {}.",
                                                          ddc->getDecimationFactor(), div, startSample));
                    }
                    startSampleRaw = startSample * (uint32_t) ddc->getDecimationFactor();
                    sampleOffset = getTxStartSampleNumberAfeDemod(ddc->getDecimationFactor());
                    nSamplesRaw = nSamples * 2;
                    sampleSize = 2 * sizeof(OutputDType);
                } else {
                    startSampleRaw = startSample * op.getRxDecimationFactor();
                    sampleOffset = TX_SAMPLE_DELAY_RAW_DATA;
                    nSamplesRaw = nSamples;
                    sampleSize = sizeof(OutputDType);
                }
                size_t nBytes = nSamples * N_RX_CHANNELS * sampleSize;

                ARRUS_REQUIRES_AT_MOST(
                    outputAddress + nBytes, DDR_SIZE,
                    ::arrus::format("Total data size cannot exceed 4GiB (device {})", getDeviceId().toString()));

                ius4oem->ScheduleReceive(firing, outputAddress, nSamplesRaw, sampleOffset + startSampleRaw,
                                         op.getRxDecimationFactor() - 1, rxMapId, nullptr);
                if (batchIdx == 0) {
                    size_t partSize = 0;
                    if(!op.isRxNOP() || this->isMaster()) {
                        partSize = nBytes;
                    }
                    // Otherwise, make an empty part (i.e. partSize = 0).
                    // (note: the firing number will be needed for transfer configuration to release element in
                    // us4oem sequencer).
                    rxBufferElementParts.emplace_back(outputAddress, partSize, firing);
                }
                if (!op.isRxNOP() || this->isMaster()) {
                    // Also, allows rx nops for master module.
                    // Master module gathers frame metadata, so we cannot miss any of it.
                    // All RX nops are just overwritten.
                    outputAddress += nBytes;
                    totalNSamples += (unsigned) nSamples;
                }
            }
        }
        // The size of the chunk, in the number of BYTES.
        auto size = outputAddress - transferAddressStart;
        // Where the chunk starts.
        auto srcAddress = transferAddressStart;
        transferAddressStart = outputAddress;
        framework::NdArray::Shape shape;
        if (isDDCOn) {
            shape = {totalNSamples, 2, N_RX_CHANNELS};
        } else {
            shape = {totalNSamples, N_RX_CHANNELS};
        }
        rxBufferElements.emplace_back(srcAddress, size, firing, shape, NdArrayDataType);
    }
    ius4oem->EnableTransmit();

    // Set frame repetition interval if possible.
    float totalPri = 0.0f;
    for (auto &op : seq) {
        totalPri += op.getPri();
    }
    std::optional<float> lastPriExtend = std::nullopt;

    // Sequence repetition interval.
    if (sri.has_value()) {
        if (totalPri < sri.value()) {
            lastPriExtend = sri.value() - totalPri;
        } else {
            // TODO move this condition to sequence validator
            throw IllegalArgumentException(format("Sequence repetition interval {} cannot be set, "
                                                  "sequence total pri is equal {}",
                                                  sri.value(), totalPri));
        }
    }

    // Program triggers
    ius4oem->SetNTriggers(nOps * batchSize * rxBufferSize);
    firing = 0;
    for (uint16 batchIdx = 0; batchIdx < rxBufferSize; ++batchIdx) {
        for (uint16 seqIdx = 0; seqIdx < batchSize; ++seqIdx) {
            for (uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
                firing = (uint16) (opIdx + seqIdx * nOps + batchIdx * batchSize * nOps);
                auto const &op = seq[opIdx];
                // checkpoint only when it is the last operation of the last batch element
                bool checkpoint = triggerSync && (opIdx == seq.size() - 1 && seqIdx == batchSize - 1);
                float pri = op.getPri();
                if (opIdx == nOps - 1 && lastPriExtend.has_value()) {
                    pri += lastPriExtend.value();
                }
                auto priMs = static_cast<unsigned int>(std::round(pri * 1e6));
                ius4oem->SetTrigger(priMs, checkpoint, firing, checkpoint && externalTrigger);
            }
        }
    }
    setAfeDemod(ddc);
    return {Us4OEMBuffer(rxBufferElements, rxBufferElementParts), std::move(fcm)};
}

float Us4OEMImpl::getTxRxTime(float rxTime) const {
    float txrxTime = 0.0f;
    if (reprogrammingMode == Us4OEMSettings::ReprogrammingMode::SEQUENTIAL) {
        txrxTime = rxTime + SEQUENCER_REPROGRAMMING_TIME;
    } else if (reprogrammingMode == Us4OEMSettings::ReprogrammingMode::PARALLEL) {
        txrxTime = std::max(rxTime, SEQUENCER_REPROGRAMMING_TIME);
    } else {
        throw IllegalArgumentException(
            format("Unrecognized reprogramming mode: {}", static_cast<size_t>(reprogrammingMode)));
    }
    return txrxTime;
}

std::tuple<std::unordered_map<uint16, uint16>, std::vector<Us4OEMImpl::Us4OEMBitMask>, FrameChannelMapping::Handle>
Us4OEMImpl::setRxMappings(const std::vector<TxRxParameters> &seq) {
    // a map: op ordinal number -> rx map id
    std::unordered_map<uint16, uint16> result;
    std::unordered_map<std::vector<uint8>, uint16, ContainerHash<std::vector<uint8>>> rxMappings;

    // FC mapping
    auto numberOfOutputFrames = getNumberOfNoRxNOPs(seq);
    if (this->isMaster()) {
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

    for (const auto &op : seq) {
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
        for (const auto isOn : op.getRxAperture()) {
            if (isOn) {
                isRxNop = false;
                ARRUS_REQUIRES_TRUE_E(onChannel < N_RX_CHANNELS,
                                      ArrusException("Up to 32 active rx channels can be set."));

                // Physical channel number, values 0-31
                auto rxChannel = channelMapping[channel];
                rxChannel = rxChannel % N_RX_CHANNELS;
                if (!setContains(channelsUsed, rxChannel) && !setContains(this->channelsMask, channel)) {
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
                if (this->isMaster()) {
                    frameNumber = opId;
                }
                fcmBuilder.setChannelMapping(frameNumber, onChannel,
                                             static_cast<FrameChannelMapping::Us4OEMNumber>(getDeviceId().getOrdinal()),
                                             frameNumber, (int8) (mapping.size() - 1));
                ++onChannel;
            }
            ++channel;
        }
        outputRxApertures.push_back(outputRxAperture);

        // Replace invalid channels with unused channels
        std::list<uint8> unusedChannels;
        for (uint8 i = 0; i < N_RX_CHANNELS; ++i) {
            if (!setContains(channelsUsed, i)) {
                unusedChannels.push_back(i);
            }
        }
        std::vector<uint8> rxMapping;
        for (auto &dstChannel : mapping) {
            if (!dstChannel.has_value()) {
                rxMapping.push_back(unusedChannels.front());
                unusedChannels.pop_front();
            } else {
                rxMapping.push_back(dstChannel.value());
            }
        }
        // Move all the non-active channels to the end of mapping.
        while (rxMapping.size() != 32) {
            rxMapping.push_back(unusedChannels.front());
            unusedChannels.pop_front();
        }

        auto mappingIt = rxMappings.find(rxMapping);
        if (mappingIt == std::end(rxMappings)) {
            // Create new Rx channel mapping.
            rxMappings.emplace(rxMapping, rxMapId);
            result.emplace(opId, rxMapId);
            // Set channel mapping
            ARRUS_REQUIRES_TRUE(rxMapping.size() == N_RX_CHANNELS,
                                format("Invalid size of the RX channel mapping to set: {}", rxMapping.size()));
            ARRUS_REQUIRES_TRUE(
                rxMapId < 128,
                format("128 different rx mappings can be loaded only, deviceId: {}.", getDeviceId().toString()));
            ius4oem->SetRxChannelMapping(rxMapping, rxMapId);
            ++rxMapId;
        } else {
            // Use the existing one.
            result.emplace(opId, mappingIt->second);
        }
        ++opId;
        if (!isRxNop) {
            ++noRxNopId;
        }
    }
    return {result, outputRxApertures, fcmBuilder.build()};
}

float Us4OEMImpl::getSamplingFrequency() { return Us4OEMImpl::SAMPLING_FREQUENCY; }

float Us4OEMImpl::getRxTime(size_t nSamples, float samplingFrequency) {
    return std::max(MIN_RX_TIME, (float) nSamples / samplingFrequency + RX_TIME_EPSILON);
}

std::bitset<Us4OEMImpl::N_ADDR_CHANNELS> Us4OEMImpl::filterAperture(std::bitset<N_ADDR_CHANNELS> aperture) {
    for (auto channel : this->channelsMask) {
        aperture[channel] = false;
    }
    return aperture;
}

void Us4OEMImpl::validateAperture(const std::bitset<N_ADDR_CHANNELS> &aperture) {
    for (auto channel : this->channelsMask) {
        if (aperture[channel]) {
            throw ArrusException(::arrus::format("Attempted to set masked channel: {}", channel));
        }
    }
}

void Us4OEMImpl::start() { this->startTrigger(); }

void Us4OEMImpl::stop() { this->stopTrigger(); }

void Us4OEMImpl::syncTrigger() { this->ius4oem->TriggerSync(); }

Ius4OEMRawHandle Us4OEMImpl::getIUs4oem() { return ius4oem.get(); }

void Us4OEMImpl::enableSequencer() {
    bool txConfOnTrigger = false;
    switch (reprogrammingMode) {
    case Us4OEMSettings::ReprogrammingMode::SEQUENTIAL: txConfOnTrigger = false; break;
    case Us4OEMSettings::ReprogrammingMode::PARALLEL: txConfOnTrigger = true; break;
    }
    this->ius4oem->EnableSequencer(txConfOnTrigger);
}

std::vector<uint8_t> Us4OEMImpl::getChannelMapping() { return channelMapping; }

// AFE setters
void Us4OEMImpl::setTgcCurve(const RxSettings &afeCfg) {
    const ops::us4r::TGCCurve &tgc = afeCfg.getTgcSamples();
    bool applyCharacteristic = afeCfg.isApplyTgcCharacteristic();

    auto tgcMax = static_cast<float>(afeCfg.getPgaGain() + afeCfg.getLnaGain());
    auto tgcMin = tgcMax - TGC_ATTENUATION_RANGE;
    // Set.
    if (tgc.empty()) {
        ius4oem->TGCDisable();
    } else {
        ius4oem->TGCEnable();
        std::vector<float> actualTgc;

        if (applyCharacteristic) {
            static const std::vector<float> tgcChar = {
                14.000f, 14.001f, 14.002f, 14.003f, 14.024f, 14.168f, 14.480f, 14.825f, 15.234f, 15.770f, 16.508f,
                17.382f, 18.469f, 19.796f, 20.933f, 21.862f, 22.891f, 24.099f, 25.543f, 26.596f, 27.651f, 28.837f,
                30.265f, 31.690f, 32.843f, 34.045f, 35.543f, 37.184f, 38.460f, 39.680f, 41.083f, 42.740f, 44.269f,
                45.540f, 46.936f, 48.474f, 49.895f, 50.966f, 52.083f, 53.256f, 54.0f};
            // observed -> applied (e.g. when applying 15 dB actually 14.01 gain was observed)
            actualTgc = ::arrus::interpolate1d(tgcChar, ::arrus::getRange<float>(14, 55, 1.0), tgc);
        } else {
            actualTgc = tgc;
        }
        for (auto &val : actualTgc) {
            val = (val - tgcMin) / TGC_ATTENUATION_RANGE;
        }
        // Currently setting firing parameter has no impact on the result
        // because TGC can be set only once for the whole sequence.
        ius4oem->TGCSetSamples(actualTgc, 0);
    }
}

void Us4OEMImpl::setRxSettings(const RxSettings &newSettings) { setRxSettingsPrivate(newSettings, false); }

void Us4OEMImpl::setRxSettingsPrivate(const RxSettings &newSettings, bool force) {
    setPgaGainAfe(newSettings.getPgaGain(), force);
    setLnaGainAfe(newSettings.getLnaGain(), force);
    setTgcCurve(newSettings);
    setDtgcAttenuationAfe(newSettings.getDtgcAttenuation(), force);
    setLpfCutoffAfe(newSettings.getLpfCutoff(), force);
    setActiveTerminationAfe(newSettings.getActiveTermination(), force);
    this->rxSettings = newSettings;
}

inline void Us4OEMImpl::setPgaGainAfe(uint16 value, bool force) {
    if (value != this->rxSettings.getPgaGain() || force) {
        ius4oem->SetPGAGain(PGAGainValueMap::getInstance().getEnumValue(value));
    }
}

inline void Us4OEMImpl::setLnaGainAfe(uint16 value, bool force) {
    if (value != this->rxSettings.getLnaGain() || force) {
        ius4oem->SetLNAGain(LNAGainValueMap::getInstance().getEnumValue(value));
    }
}

inline void Us4OEMImpl::setDtgcAttenuationAfe(std::optional<uint16> param, bool force) {
    if (param == rxSettings.getDtgcAttenuation() && !force) {
        return;
    }
    if (param.has_value()) {
        ius4oem->SetDTGC(us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_EN,
                         DTGCAttenuationValueMap::getInstance().getEnumValue(param.value()));
    } else {
        // DTGC param does not matter
        ius4oem->SetDTGC(us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_DIS,
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_42dB);
    }
}

inline void Us4OEMImpl::setLpfCutoffAfe(uint32 value, bool force) {
    if (value != this->rxSettings.getLpfCutoff() || force) {
        ius4oem->SetLPFCutoff(LPFCutoffValueMap::getInstance().getEnumValue(value));
    }
}

inline void Us4OEMImpl::setActiveTerminationAfe(std::optional<uint16> param, bool force) {
    if (param == rxSettings.getActiveTermination() && !force) {
        return;
    }
    if (rxSettings.getActiveTermination().has_value()) {
        ius4oem->SetActiveTermination(us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_EN,
                                      ActiveTerminationValueMap::getInstance().getEnumValue(param.value()));
    } else {
        ius4oem->SetActiveTermination(us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_DIS,
                                      us4r::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50);
    }
}

float Us4OEMImpl::getFPGATemperature() { return ius4oem->GetFPGATemp(); }

float Us4OEMImpl::getUCDMeasuredVoltage(uint8_t rail) { return ius4oem->GetUCDVOUT(rail); }

void Us4OEMImpl::checkFirmwareVersion() {
    try {
        ius4oem->CheckFirmwareVersion();
    } catch (const std::runtime_error &e) { throw arrus::IllegalStateException(e.what()); } catch (...) {
        throw arrus::IllegalStateException("Unknown exception while check firmware version.");
    }
}

uint32 Us4OEMImpl::getFirmwareVersion() { return ius4oem->GetFirmwareVersion(); }

uint32 Us4OEMImpl::getTxFirmwareVersion() { return ius4oem->GetTxFirmwareVersion(); }

void Us4OEMImpl::checkState() { this->checkFirmwareVersion(); }

void Us4OEMImpl::setTestPattern(RxTestPattern pattern) {
    switch (pattern) {
    case RxTestPattern::RAMP: ius4oem->EnableTestPatterns(); break;
    case RxTestPattern::OFF: ius4oem->DisableTestPatterns(); break;
    default: throw IllegalArgumentException("Unrecognized test pattern");
    }
}

uint32_t Us4OEMImpl::getTxStartSampleNumberAfeDemod(float ddcDecimationFactor) const {
    //DDC valid data offset
    uint32_t offset = 34u + (16 * (uint32_t) ddcDecimationFactor);

    //Check if data valid offset is higher than TX offset
    if (offset > 240) {
        //If TX offset is lower than data valid offset return just data valid offset and log warning
        this->logger->log(LogSeverity::WARNING,
                          ::arrus::format("Decimation factor {} causes RX data to start after the moment TX starts."
                                          " Delay TX by {} cycles to align start of RX data with start of TX.",
                                          ddcDecimationFactor, (offset - 240)));
        return offset;
    } else {
        //Calculate offset pointing to DDC sample closest but lower than 240 cycles (TX offset)
        if(ddcDecimationFactor == 4) {
            // Note: for some reason us4OEM AFE has a different offset for
            // decimation factor; the below values was determined
            // experimentally.
            return offset + 2*84;
        }
        else {
            offset += ((240u - offset) / (uint32_t) ddcDecimationFactor) * (uint32_t) ddcDecimationFactor;
            return offset;
        }
    }
}

float Us4OEMImpl::getCurrentSamplingFrequency() const {
    std::unique_lock<std::mutex> lock{stateMutex};
    return currentSamplingFrequency;
}

float Us4OEMImpl::getFPGAWallclock() { return ius4oem->GetFPGAWallclock(); }

void Us4OEMImpl::setAfeDemod(const std::optional<ops::us4r::DigitalDownConversion> &ddc) {
    if (ddc.has_value()) {
        auto &value = ddc.value();
        setAfeDemod(value.getDemodulationFrequency(), value.getDecimationFactor(), value.getFirCoefficients().data(),
                    value.getFirCoefficients().size());
    }
}

void Us4OEMImpl::setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                             size_t nCoefficients) {
    //check decimation factor
    if (!(decimationFactor >= 2.0f && decimationFactor <= 63.75f)) {
        throw IllegalArgumentException("Decimation factor should be in range 2.0 - 63.75");
    }

    int decInt = static_cast<int>(decimationFactor);
    float decFract = decimationFactor - static_cast<float>(decInt);
    int nQuarters = 0;
    if (decFract == 0.0f || decFract == 0.25f || decFract == 0.5f || decFract == 0.75f) {
        nQuarters = int(decFract * 4.0f);
    } else {
        throw IllegalArgumentException("Decimation's fractional part should be equal 0.0, 0.25, 0.5 or 0.75");
    }
    int expectedNumberOfCoeffs = 0;
    //check if fir size is correct for given decimation factor
    if (nQuarters == 0) {
        expectedNumberOfCoeffs = 8 * decInt;
    } else if (nQuarters == 1) {
        expectedNumberOfCoeffs = 32 * decInt + 8;
    } else if (nQuarters == 2) {
        expectedNumberOfCoeffs = 16 * decInt + 8;
    } else if (nQuarters == 3) {
        expectedNumberOfCoeffs = 32 * decInt + 24;
    }
    if (static_cast<size_t>(expectedNumberOfCoeffs) != nCoefficients) {
        throw IllegalArgumentException(format("Incorrect number of DDC FIR filter coefficients, should be {}, "
                                              "actual: {}",
                                              expectedNumberOfCoeffs, nCoefficients));
    }
    enableAfeDemod();
    setAfeDemodConfig(static_cast<uint8_t>(decInt), static_cast<uint8_t>(nQuarters), firCoefficients,
                      static_cast<uint16_t>(nCoefficients), demodulationFrequency);
}

uint8_t Us4OEMImpl::getStopBits() { return ius4oem->GetStopBits(); }

}// namespace arrus::devices
