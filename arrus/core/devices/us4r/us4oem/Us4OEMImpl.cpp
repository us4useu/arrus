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
                       bool externalTrigger = false, bool acceptRxNops = false)
    : Us4OEMImplBase(id), logger{getLoggerFactory()->getLogger()}, ius4oem(std::move(ius4oem)),
      channelMapping(std::move(channelMapping)), channelsMask(std::move(channelsMask)),
      reprogrammingMode(reprogrammingMode), rxSettings(std::move(rxSettings)), externalTrigger(externalTrigger),
      serialNumber([this](){return this->ius4oem->GetSerialNumber();}),
      revision([this](){return this->ius4oem->GetRevisionNumber();}),
      acceptRxNops(acceptRxNops) {

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
    Us4OEMTxRxValidator(const std::string &componentName, float txFrequencyMin, float txFrequencyMax, std::optional<float> maxPulseLength=std::nullopt)
        : Validator(componentName), txFrequencyMin(txFrequencyMin), txFrequencyMax(txFrequencyMax), maxPulseLength(maxPulseLength) {}

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
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getTxPulse().getCenterFrequency(), txFrequencyMin, txFrequencyMax,
                                                  firingStr);
                if(maxPulseLength.has_value()) {
                    float pulseLength = op.getTxPulse().getNPeriods()/op.getTxPulse().getCenterFrequency();
                    ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(pulseLength, 0.0f, maxPulseLength.value(), firingStr);
                }
                else {
                    // The legacy OEM constraint
                    ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getTxPulse().getNPeriods(), 0.0f, 32.0f, firingStr);
                }
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

private:
    float txFrequencyMin;
    float txFrequencyMax;
    std::optional<float> maxPulseLength;
};


std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
Us4OEMImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgc, uint16 rxBufferSize,
                            uint16 batchSize, std::optional<float> sri, arrus::ops::us4r::Scheme::WorkMode workMode,
                            const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc,
                            const std::vector<arrus::framework::NdArray> &txDelays) {
    std::unique_lock<std::mutex> lock{stateMutex};
    // Validate input sequence and parameters.
    std::string deviceIdStr = getDeviceId().toString();
    bool isDDCOn = ddc.has_value();
    Us4OEMTxRxValidator seqValidator(
        format("{} tx rx sequence", deviceIdStr),
        ius4oem->GetMinTxFrequency(),
        ius4oem->GetMaxTxFrequency(),
        this->maxPulseLength
    );
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
    ius4oem->ResetSequencer();
    ius4oem->SetNumberOfFirings(nOps);
    ius4oem->ClearScheduledReceive();
    ius4oem->ResetCallbacks();
    auto [rxMappings, rxApertures, fcm] = setRxMappings(seq);
    // helper data
    const std::bitset<N_ADDR_CHANNELS> emptyAperture;
    const std::bitset<N_ACTIVE_CHANNEL_GROUPS> emptyChannelGroups;
    this->isDecimationFactorAdjustmentLogged = false;

    size_t nTxDelayProfiles = txDelays.size();

    bool triggerSyncPerBatch = arrus::ops::us4r::Scheme::isWorkModeManual(workMode) || workMode == ops::us4r::Scheme::WorkMode::HOST;
    bool triggerSyncPerTxRx = workMode == ops::us4r::Scheme::WorkMode::MANUAL_OP;


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
        size_t currentTxDelaysId = 0;
        uint8 txChannel = 0;
        for (bool bit : op.getTxAperture()) {
            // First set the internal TX delays.
            for(currentTxDelaysId = 0; currentTxDelaysId < nTxDelayProfiles; ++currentTxDelaysId) {
                float txDelay = 0.0f;
                if (bit && !::arrus::setContains(this->channelsMask, txChannel)) {
                    txDelay = txDelays[currentTxDelaysId].get<float>((size_t)opIdx, (size_t)txChannel);
                }
                ius4oem->SetTxDelay(txChannel, txDelay, opIdx, currentTxDelaysId);
            }
            // Then set the profile from the input sequence (for backward-compatibility).
            // NOTE: this might look redundant and it is, however it simplifies the changes for v0.9.0 a lot
            // and reduces the risk of causing new bugs in the whole mapping implementation.
            // This will be optimized in v0.10.0.
            float txDelay = 0.0f;
            if (bit && !::arrus::setContains(this->channelsMask, txChannel)) {
                txDelay = op.getTxDelays()[txChannel];
            }
            ius4oem->SetTxDelay(txChannel, txDelay, opIdx, currentTxDelaysId);
            ++txChannel;
        }
        ius4oem->SetTxFreqency(op.getTxPulse().getCenterFrequency(), opIdx);
        ius4oem->SetTxHalfPeriods(static_cast<uint32>(op.getTxPulse().getNPeriods() * 2), opIdx);
        ius4oem->SetTxInvert(op.getTxPulse().isInverse(), opIdx);
        ius4oem->SetRxTime(rxTime, opIdx);
        ius4oem->SetRxDelay(op.getRxDelay(), opIdx);
    }
    // Set the last profile as the current TX delay (the last one is the one provided in the Sequence.ops.Tx.delays property.
    ius4oem->SetTxDelays(nTxDelayProfiles);
    // NOTE: for us4OEM+ the method below must be called right after programming TX/RX, and before calling ScheduleReceive.
    ius4oem->SetNTriggers(nOps * batchSize * rxBufferSize);

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
                    sampleOffset = ius4oem->GetTxOffset();
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
                    unsigned partNSamples = 0;
                    if(!op.isRxNOP() || acceptRxNops) {
                        partSize = nBytes;
                        partNSamples = (unsigned) nSamples;
                    }
                    // Otherwise, make an empty part (i.e. partSize = 0).
                    // (note: the firing number will be needed for transfer configuration to release element in
                    // us4oem sequencer, and for the subSequence setter).
                    rxBufferElementParts.emplace_back(outputAddress, partSize, firing, partNSamples);
                }
                if (!op.isRxNOP() || acceptRxNops) {
                    // Also, allows rx nops.
                    // For example, the master module gathers frame metadata, so we cannot miss any of it.
                    // In all other cases, all RX nops are just overwritten.
                    outputAddress += nBytes;
                    totalNSamples += (unsigned) nSamples;
                }
            }
        }
        // The size of the chunk, in the number of BYTES.
        // NOTE: THE BELOW LINE MUST BE CONSISTENT WITH Us4OEMBuffer::getView IMPLEMENTATION!
        auto size = outputAddress - transferAddressStart;
        // Where the chunk starts.
        auto srcAddress = transferAddressStart;
        transferAddressStart = outputAddress;
        // NOTE: THE BELOW LINE MUST BE CONSISTENT WITH Us4OEMBuffer::getView IMPLEMENTATION!
        framework::NdArray::Shape shape = Us4OEMBuffer::getShape(isDDCOn, totalNSamples, N_RX_CHANNELS);
        rxBufferElements.emplace_back(srcAddress, size, firing, shape, NdArrayDataType);
    }

    // Set frame repetition interval if possible.
    std::optional<float> lastPriExtend = getLastPriExtend(
        std::begin(seq), std::end(seq), sri
    );

    // Program triggers
    firing = 0;
    for (uint16 batchIdx = 0; batchIdx < rxBufferSize; ++batchIdx) {
        for (uint16 seqIdx = 0; seqIdx < batchSize; ++seqIdx) {
            for (uint16 opIdx = 0; opIdx < seq.size(); ++opIdx) {
                firing = (uint16) (opIdx + seqIdx * nOps + batchIdx * batchSize * nOps);
                auto const &op = seq[opIdx];
                // checkpoint only when it is the last operation of the last batch element
                bool checkpoint = triggerSyncPerBatch && (opIdx == seq.size() - 1 && seqIdx == batchSize - 1);
                float pri = op.getPri();
                if (opIdx == nOps - 1 && lastPriExtend.has_value()) {
                    pri += lastPriExtend.value();
                }
                auto priMs = getTimeToNextTrigger(pri);
                ius4oem->SetTrigger(
                    priMs,
                    checkpoint || triggerSyncPerTxRx,
                    firing,
                    checkpoint && externalTrigger,
                    triggerSyncPerTxRx
                );
            }
        }
    }
    setAfeDemod(ddc);
    this->currentSequence = seq;

    if(arrus::ops::us4r::Scheme::isWorkModeManual(workMode)) {
        // Register event_done callback in case we would like to wait for the interrupt to happen
        auto eventDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::EVENTDONE);
        irqsRegistered.at(eventDoneIrq) = 0;
        irqsHandled.at(eventDoneIrq) = 0;
        ius4oem->RegisterCallback(IUs4OEM::MSINumber::EVENTDONE, [eventDoneIrq, this]() {
            std::unique_lock l(irqEventMutex.at(eventDoneIrq));
            ++(irqsRegistered.at(eventDoneIrq));
            irqEvent.at(eventDoneIrq).notify_one();
        });
    }

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
    if (acceptRxNops) {
        // We transfer all module frames due to possible metadata stored in the frame (if enabled).
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
                if (acceptRxNops) {
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

void Us4OEMImpl::sync(std::optional<long long> timeout) {
    logger->log(LogSeverity::TRACE, "Waiting for EVENTDONE IRQ");
    auto eventDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::EVENTDONE);
    this->waitForIrq(eventDoneIrq, timeout);
}

void Us4OEMImpl::setWaitForHVPSMeasurementDone() {
    ius4oem->EnableHVPSMeasurementReadyIRQ();
    auto measurementDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE);
    irqsRegistered.at(measurementDoneIrq) = 0;
    irqsHandled.at(measurementDoneIrq) = 0;
    ius4oem->RegisterCallback(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE, [measurementDoneIrq, this]() {
        std::unique_lock l(irqEventMutex.at(measurementDoneIrq));
        ++(irqsRegistered.at(measurementDoneIrq));
        irqEvent.at(measurementDoneIrq).notify_one();
    });
}

void Us4OEMImpl::waitForHVPSMeasurementDone(std::optional<long long> timeout) {
    logger->log(LogSeverity::TRACE, "Waiting for HVPS Measurement done IRQ");
    auto measurementDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE);
    this->waitForIrq(measurementDoneIrq, timeout);
}

Ius4OEMRawHandle Us4OEMImpl::getIUs4oem() { return ius4oem.get(); }

void Us4OEMImpl::enableSequencer(bool resetSequencerPointer) {
    bool txConfOnTrigger = false;
    switch (reprogrammingMode) {
    case Us4OEMSettings::ReprogrammingMode::SEQUENTIAL: txConfOnTrigger = false; break;
    case Us4OEMSettings::ReprogrammingMode::PARALLEL: txConfOnTrigger = true; break;
    }
    this->ius4oem->EnableSequencer(txConfOnTrigger, resetSequencerPointer);
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
        std::vector<float> actualTgc = tgc;
        // Normalize to [0, 1].
        for (auto &val : actualTgc) {
            val = (val - tgcMin) / TGC_ATTENUATION_RANGE;
        }
        if (applyCharacteristic) {
            // TGC characteristic, experimentally verified.
            static const std::vector<float> tgcChar = {
                0.0f, 2.4999999999986144e-05f, 5.00000000000167e-05f, 7.500000000000284e-05f,
                0.0005999999999999784f, 0.0041999999999999815f, 0.01200000000000001f, 0.020624999999999984f,
                0.03085f, 0.04424999999999999f, 0.06269999999999998f, 0.08455000000000004f,
                0.11172500000000003f, 0.14489999999999997f, 0.173325f, 0.19654999999999995f,
                0.22227499999999994f, 0.252475f, 0.28857499999999997f, 0.3149f,
                0.341275f, 0.370925f, 0.406625f, 0.44225000000000003f,
                0.4710750000000001f, 0.501125f, 0.538575f, 0.5795999999999999f,
                0.6115f, 0.642f, 0.677075f, 0.7185f,
                0.756725f, 0.7885f, 0.8234f, 0.8618499999999999f,
                0.897375f, 0.92415f, 0.952075f, 0.9814f, 1.0f
            };
            // the below is simply linspace(0, 1, 41)
            static const std::vector<float> tgcCharPoints = {
                0.0f   , 0.025f, 0.05f, 0.075f, 0.1f, 0.125f, 0.15f, 0.175f, 0.2f  ,
                0.225f, 0.25f , 0.275f, 0.3f  , 0.325f, 0.35f , 0.375f, 0.4f  , 0.425f,
                0.45f , 0.475f, 0.5f  , 0.525f, 0.55f , 0.575f, 0.6f  , 0.625f, 0.65f ,
                0.675f, 0.7f  , 0.725f, 0.75f , 0.775f, 0.8f  , 0.825f, 0.85f , 0.875f,
                0.9f  , 0.925f, 0.95f , 0.975f, 1.0f
            };
            actualTgc = ::arrus::interpolate1d(tgcChar, tgcCharPoints, actualTgc);
        }
        ius4oem->TGCEnable();
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
        ius4oem->SetDTGC(::us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_EN,
                         DTGCAttenuationValueMap::getInstance().getEnumValue(param.value()));
    } else {
        // DTGC param does not matter
        ius4oem->SetDTGC(::us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_DIS,
                         ::us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_42dB);
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
        ius4oem->SetActiveTermination(::us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_EN,
                                      ActiveTerminationValueMap::getInstance().getEnumValue(param.value()));
    } else {
        ius4oem->SetActiveTermination(::us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_DIS,
                                      ::us4r::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50);
    }
}

float Us4OEMImpl::getFPGATemperature() { return ius4oem->GetFPGATemp(); }

float Us4OEMImpl::getUCDTemperature() { return ius4oem->GetUCDTemp(); }

float Us4OEMImpl::getUCDExternalTemperature() { return ius4oem->GetUCDExtTemp(); }

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

uint32_t Us4OEMImpl::getTxOffset()  { return ius4oem->GetTxOffset(); }

uint32_t Us4OEMImpl::getOemVersion()  { return ius4oem->GetOemVersion(); }

void Us4OEMImpl::checkState() { this->checkFirmwareVersion(); }

void Us4OEMImpl::setTestPattern(RxTestPattern pattern) {
    switch (pattern) {
    case RxTestPattern::RAMP: ius4oem->EnableTestPatterns(); break;
    case RxTestPattern::OFF: ius4oem->DisableTestPatterns(); break;
    default: throw IllegalArgumentException("Unrecognized test pattern");
    }
}

uint32_t Us4OEMImpl::getTxStartSampleNumberAfeDemod(float ddcDecimationFactor) {
    //DDC valid data offset
    uint32_t txOffset = ius4oem->GetTxOffset();
    uint32_t offset = 34u + (uint32_t)(16 * ddcDecimationFactor);
    uint32_t offsetCorrection = 0;

    float decInt = 0;
    float decFloat = modf(ddcDecimationFactor, &decInt);

    uint32_t dataStep = (uint32_t)decInt;
    if (decFloat == 0.5f) {
        dataStep = (uint32_t)(2.0f * ddcDecimationFactor);
    } else if (decFloat == 0.25f || decFloat == 0.75f) {
        dataStep = (uint32_t)(4.0f * ddcDecimationFactor);
    }

     if(ddcDecimationFactor == 4.0f) {
        // Note: for some reason us4OEM AFE has a different offset for
        // decimation factor = 4; the below value was determined
        // experimentally (TX starts at 266 RX sample offset).
        offsetCorrection = (4 * 7);
    }

    //Check if data valid offset is higher than TX offset
    if (offset > txOffset) {
        //If TX offset is lower than data valid offset return just data valid offset and log warning
        if(!this->isDecimationFactorAdjustmentLogged) {
            this->logger->log(LogSeverity::INFO,
                          ::arrus::format("Decimation factor {} causes RX data to start after the moment TX starts."
                                          " Delay TX by {} microseconds to align start of RX data with start of TX.",
                                          ddcDecimationFactor, (float)(offset - txOffset + offsetCorrection)/65.0f));
            this->isDecimationFactorAdjustmentLogged = true;
        }
        return offset;
    } else {
        //Calculate offset pointing to DDC sample closest but lower than 240 cycles (TX offset)
        offset += ((txOffset - offset) / dataStep) * dataStep;

        return (offset + offsetCorrection);
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
    else {
        disableAfeDemod();
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

const char* Us4OEMImpl::getSerialNumber() { return this->serialNumber.get().c_str(); }

const char *Us4OEMImpl::getRevision() { return this->revision.get().c_str(); }

void Us4OEMImpl::setSubsequence(uint16 start, uint16 end, bool syncMode, const std::optional<float> &sri) {
    // NOTE: end is inclusive (and the below method expects [start, end) range.
    std::optional<float> priExtend = getLastPriExtend(
        std::begin(currentSequence)+start,
        std::begin(currentSequence)+end+1,
        sri
    );
    uint32_t timeToNextTrigger = 0;
    if(priExtend.has_value()) {
        timeToNextTrigger = getTimeToNextTrigger(priExtend.value()+this->currentSequence.at(end).getPri());
    }
    else {
        // Just use the PRI of the end TX/RX.
        timeToNextTrigger = getTimeToNextTrigger(this->currentSequence.at(end).getPri());
    }
    this->ius4oem->SetSubsequence(start, end, syncMode, timeToNextTrigger);
}

void Us4OEMImpl::clearCallbacks() {
    this->ius4oem->ClearCallbacks();
}

void Us4OEMImpl::waitForIrq(unsigned int irq, std::optional<long long> timeout) {
    std::unique_lock lock(irqEventMutex.at(irq));
    if(timeout.has_value()) {
        bool isReady = irqEvent.at(irq).wait_for(lock, std::chrono::milliseconds(timeout.value()),
            [irq, this]() {
                // Wait until the number of registered interrupts is greater than the number of IRQs already handled.
                // (i.e. there is some new, unhandled interrupt).
                return this->irqsRegistered.at(irq) > this->irqsHandled.at(irq);
            });
        if(!isReady) {
            throw TimeoutException("Timeout on waiting for trigger to be registered. Is the system still alive?");
        }
    }
    else {
        // No timeout, wait infinitely.
        irqEvent.at(irq).wait(lock, [irq, this]() {
            return this->irqsRegistered.at(irq) > this->irqsHandled.at(irq);
        } );

    }
    if(this->irqsRegistered.at(irq) != this->irqsHandled.at(irq)+1) {
        // In the correct scenario, we expect that the number of already handled IRQs is equal to the number of
        // registered IRQs minus 1.
        // If it's not true, it means that we have lost some IRQ -- this is an exception that user should react to.
        throw IllegalStateException(format("The number of registered IRQs is different than the number of handled IRQs."
                                    " We detected missing {} IRQs.", irq));
    }
    ++this->irqsHandled.at(irq);
}

HVPSMeasurement Us4OEMImpl::getHVPSMeasurement() {
    auto m = ius4oem->GetHVPSMeasurements();
    HVPSMeasurementBuilder builder;
    builder.set(0, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::VOLTAGE, m.HVP0Voltage);
    builder.set(0, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::CURRENT, m.HVP0Current);
    builder.set(1, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::VOLTAGE, m.HVP1Voltage);
    builder.set(1, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::CURRENT, m.HVP1Current);
    builder.set(0, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::VOLTAGE, m.HVM0Voltage);
    builder.set(0, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::CURRENT, m.HVM0Current);
    builder.set(1, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::VOLTAGE, m.HVM1Voltage);
    builder.set(1, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::CURRENT, m.HVM1Current);
    return builder.build();
}

float Us4OEMImpl::setHVPSSyncMeasurement(uint16_t nSamples, float frequency) {
    return ius4oem->SetHVPSSyncMeasurement(nSamples, frequency);
}

void Us4OEMImpl::setMaximumPulseLength(std::optional<float> maxLength) {
    // 2 means OEM+
    // this is the only type of OEM that currently can have a maxLength != nullopt
    if(ius4oem->GetOemVersion() != 2 && maxLength.has_value()) {
        throw IllegalArgumentException("Currently it is possible to set maxLength value only for OEM+ (type 2)");
    }
    this->maxPulseLength = maxLength;
}

}// namespace arrus::devices
