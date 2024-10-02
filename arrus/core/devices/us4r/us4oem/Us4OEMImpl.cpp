#include "Us4OEMImpl.h"

#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

#include "Us4OEMDescriptorFactory.h"
#include "Us4OEMTxRxValidator.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/common/utils.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/common/interpolate.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMRxMappingRegisterBuilder.h"

namespace arrus::devices {
// TODO migrate this source to us4r subspace

using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem, std::vector<uint8_t> channelMapping, RxSettings rxSettings,
                       Us4OEMSettings::ReprogrammingMode reprogrammingMode, Us4OEMDescriptor descriptor,
                       bool externalTrigger = false, bool acceptRxNops = false)
    : Us4OEMImplBase(id), logger{getLoggerFactory()->getLogger()}, ius4oem(std::move(ius4oem)),
      descriptor(std::move(descriptor)),
      channelMapping(std::move(channelMapping)), reprogrammingMode(reprogrammingMode),
      rxSettings(std::move(rxSettings)), externalTrigger(externalTrigger),
      serialNumber([this]() { return this->ius4oem->GetSerialNumber(); }),
      revision([this]() { return this->ius4oem->GetRevisionNumber(); }), acceptRxNops(acceptRxNops) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    setTestPattern(RxTestPattern::OFF);
    disableAfeDemod();
    ius4oem->SetRxSettings(this->rxSettings, true);
    setCurrentSamplingFrequency(this->descriptor.getSamplingFrequency());
}

Us4OEMImpl::~Us4OEMImpl() {
    try {
        logger->log(LogSeverity::DEBUG, arrus::format("Destroying handle"));
    } catch (const std::exception &e) {
        std::cerr << arrus::format("Exception while calling us4oem destructor: {}", e.what()) << std::endl;
    }
    logger->log(LogSeverity::DEBUG, arrus::format("Us4OEM handle destroyed."));
}

bool Us4OEMImpl::isMaster() { return descriptor.isMaster(); }

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

void Us4OEMImpl::setLnaHpfCornerFrequency(uint32_t frequency) {
    uint8_t coefficient = 0;
    switch (frequency) {
    case 20'000: coefficient = 8; break;
    case 50'000: coefficient = 15; break;
    case 100'000: coefficient = 0; break;
    default:
        throw ::arrus::IllegalArgumentException(::arrus::format("Unsupported LNA HPF corner frequency: {}", frequency));
    }
    ius4oem->AfeEnableLnaHPF();
    ius4oem->AfeSetLnaHPFCornerFrequency(coefficient);
}

void Us4OEMImpl::disableLnaHpf() { ius4oem->AfeDisableLnaHPF(); }

void Us4OEMImpl::setAdcHpfCornerFrequency(uint32_t frequency) {
    uint8_t coefficient = 0;
    switch (frequency) {
    case 150'000: coefficient = 4; break;
    case 300'000: coefficient = 3; break;
    case 600'000: coefficient = 2; break;
    case 1'200'000: coefficient = 1; break;
    case 2'400'000: coefficient = 0; break;
    default:
        throw ::arrus::IllegalArgumentException(::arrus::format("Unsupported ADC HPF corner frequency: {}", frequency));
    }
    ius4oem->AfeEnableAdcHPF();
    ius4oem->AfeSetAdcHPFParamsPreset(coefficient);
}

void Us4OEMImpl::disableAdcHpf() { ius4oem->AfeDisableAdcHPF(); }

Interval<Voltage> Us4OEMImpl::getAcceptedVoltageRange() { return Interval<Voltage>{0, 90}; }

void Us4OEMImpl::resetAfe() { ius4oem->AfeSoftReset(); }

Us4OEMUploadResult Us4OEMImpl::upload(const std::vector<us4r::TxRxParametersSequence> &sequences, uint16 rxBufferSize,
                                      ops::us4r::Scheme::WorkMode workMode,
                                      const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                                      const std::vector<arrus::framework::NdArray> &txDelays,
                                      const std::vector<TxTimeout> &txTimeouts) {
    std::unique_lock<std::mutex> lock{stateMutex};
    validate(sequences, rxBufferSize);
    setTgcCurve(sequences);
    ius4oem->ResetSequencer();
    ius4oem->SetNumberOfFirings(ARRUS_SAFE_CAST(getNumberOfFirings(sequences), uint16_t));
    ius4oem->ClearScheduledReceive();
    ius4oem->ResetCallbacks();
    auto rxMappingRegister = setRxMappings(sequences);
    this->isDecimationFactorAdjustmentLogged = false;
    setTxTimeouts(txTimeouts);
    uploadFirings(sequences, ddc, txDelays, rxMappingRegister);
    // For us4OEM+ the method below must be called right after programming TX/RX, and before calling ScheduleReceive.
    ius4oem->SetNTriggers(ARRUS_SAFE_CAST(getNumberOfTriggers(sequences, rxBufferSize), uint16_t));
    auto bufferDef = uploadAcquisition(sequences, rxBufferSize, ddc, rxMappingRegister);
    uploadTriggersIOBS(sequences, rxBufferSize, workMode);
    setAfeDemod(ddc);
    if(workMode == ops::us4r::Scheme::WorkMode::MANUAL_OP) {
        setWaitForEventDone();
    }
    return Us4OEMUploadResult{bufferDef, rxMappingRegister.acquireFCMs()};
}
void Us4OEMImpl::setTxTimeouts(const std::vector<TxTimeout> &txTimeouts) {
    if(!txTimeouts.empty()) {
        ius4oem->EnableTxTimeout();
        for(size_t n = 0; n < txTimeouts.size(); ++n) {
            ius4oem->SetTxTimeout((uint8_t)n, txTimeouts[n]);
        }
    }
}

Us4OEMImpl::Us4OEMChannelsGroupsMask Us4OEMImpl::getActiveChannelGroups(const Us4OEMAperture &txAperture,
                                                                        const Us4OEMAperture &rxAperture) {
    std::vector<bool> result(Us4OEMDescriptor::N_ADDR_CHANNELS, false);
    const auto &mapping = getChannelMapping();
    for (ChannelIdx logicalCh = 0; logicalCh < Us4OEMDescriptor::N_ADDR_CHANNELS; ++logicalCh) {
        if (txAperture.test(logicalCh) || rxAperture.test(logicalCh)) {
            ChannelIdx physicalCh = mapping[logicalCh];
            ChannelIdx groupNr = physicalCh / descriptor.getActiveChannelGroupSize();
            result[groupNr] = true;
        }
    }
    static const std::vector<ChannelIdx> acgRemap = {0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15};
    auto acg = permute(result, acgRemap);
    return ::arrus::toBitset<Us4OEMDescriptor::N_ACTIVE_CHANNEL_GROUPS>(acg);
}

void Us4OEMImpl::uploadFirings(const TxParametersSequenceColl &sequences,
                               const std::optional<DigitalDownConversion> &ddc,
                               const std::vector<arrus::framework::NdArray> &txDelays,
                               const Us4OEMRxMappingRegister &rxMappingRegister) {
    using SequenceId = uint16;
    using OpId = uint16;

    bool isDDCOn = ddc.has_value();
    const Us4OEMChannelsGroupsMask emptyChannelGroups;
    // us4OEM sequencer firing/entry id (global).
    OpId firingId = 0;
    for (SequenceId sequenceId = 0; sequenceId < ARRUS_SAFE_CAST(sequences.size(), SequenceId); ++sequenceId) {
        auto const &sequence = sequences[sequenceId];
        for (OpId opId = 0; opId < ARRUS_SAFE_CAST(sequence.size(), OpId); ++opId, ++firingId) {
            auto const &op = sequence.at(opId);
            logger->log(LogSeverity::TRACE,
                        format("Setting sequence {}, TX/RX {}: NOP? {}, definition: {}", sequenceId, opId, op.isNOP(),
                               ::arrus::toString(op)));
            // TX
            auto txAperture = arrus::toBitset<Us4OEMDescriptor::N_TX_CHANNELS>(op.getTxAperture());
            auto nTxHalfPeriods = static_cast<uint32>(op.getTxPulse().getNPeriods() * 2);
            // RX
            auto rxAperture = rxMappingRegister.getRxAperture(sequenceId, opId);
            float decimationFactor = isDDCOn ? ddc->getDecimationFactor() : (float) op.getRxDecimationFactor();
            setCurrentSamplingFrequency(descriptor.getSamplingFrequency() / decimationFactor);
            float rxTime = getRxTime(op, this->currentSamplingFrequency);
            // Common
            float txrxTime = getTxRxTime(rxTime);
            auto filteredTxAperture = filterAperture(txAperture, op.getMaskedChannelsTx());
            auto filteredRxAperture = filterAperture(rxAperture, op.getMaskedChannelsRx());
            Us4OEMChannelsGroupsMask channelsGroups =
                op.isNOP() ? emptyChannelGroups : getActiveChannelGroups(filteredTxAperture, filteredRxAperture);
            ARRUS_REQUIRES_TRUE_IAE(txrxTime <= op.getPri(),
                                    format("Total time required for a single TX/RX ({}) should not exceed PRI ({})",
                                           txrxTime, op.getPri()));
            // Upload
            ius4oem->SetActiveChannelGroup(channelsGroups, firingId);
            ius4oem->SetTxAperture(filteredTxAperture, firingId);
            ius4oem->SetRxAperture(filteredRxAperture, firingId);
            ius4oem->SetRxDelay(op.getRxDelay(), firingId);
            // Delays
            // Set delay defintion tables.
            for (size_t delaysId = 0; delaysId < txDelays.size(); ++delaysId) {
                auto delays = txDelays.at(delaysId).row(opId).toVector<float>();
                setTxDelays(op.getTxAperture(), delays, firingId, delaysId, op.getMaskedChannelsTx());
            }
            // Then set the profile from the input sequence (for backward-compatibility).
            // NOTE: this might look redundant and it is, however it simplifies the changes for v0.9.0 a lot
            // and reduces the risk of causing new bugs in the whole mapping implementation.
            // This will be optimized in TODO(0.12.0).
            setTxDelays(op.getTxAperture(), op.getTxDelays(), firingId, txDelays.size(),
                        op.getMaskedChannelsTx());
            ius4oem->SetTxFreqency(op.getTxPulse().getCenterFrequency(), firingId);
            ius4oem->SetTxHalfPeriods(nTxHalfPeriods, firingId);
            ius4oem->SetTxInvert(op.getTxPulse().isInverse(), firingId);
            if(isOEMPlus()) {
                ius4oem->SetTxVoltageLevel(op.getTxPulse().getAmplitudeLevel(), firingId);
            }
            ius4oem->SetRxTime(rxTime, firingId);
            if(isOEMPlus() && op.getTxTimeoutId().has_value()) {
                ius4oem->SetFiringTxTimoutId(firingId, op.getTxTimeoutId().value());
            }
        }
    }
    // Set the last profile as the current TX delay
    // (the last one is the one provided in the Sequence.ops.Tx.delays property).
    ius4oem->SetTxDelays(txDelays.size());

    // Build sequence waveform.
    for (OpId firing = 0; firing < ARRUS_SAFE_CAST(sequences.size(), OpId); ++firing) {
        ius4oem->BuildSequenceWaveform(firing);
    }
}

size_t Us4OEMImpl::scheduleReceiveDDC(size_t outputAddress, uint32 startSample, uint32 endSample, uint16 entryId,
                                      const TxRxParameters &op, uint16 rxMapId,
                                      const std::optional<DigitalDownConversion> &ddc) {
    float decInt = 0;
    float decFloat = modf(ddc->getDecimationFactor(), &decInt);

    uint32 div = 1;
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
    // Start sample, after transforming to the system number of cycles.
    // The start sample should be provided to the us4r-api
    // as for the nominal sampling frequency of us4OEM, i.e. 65 MHz.
    const uint32 startSampleRaw = startSample * (uint32_t) ddc->getDecimationFactor();
    // RX offset to the moment tx delay = 0.
    const uint32 sampleOffset = getTxStartSampleNumberAfeDemod(ddc->getDecimationFactor());
    // Number of samples to acquire per channel.
    const size_t nSamples = endSample - startSample;
    // Number of samples to be set on IUs4OEM::ScheduleReceive
    const size_t nSamplesRaw = nSamples * 2;
    // Number of bytes a single sample takes (e.g. RF: a single int16, IQ: a pair of int16)
    const size_t sampleSize = 2 * sizeof(RawDataType);
    const size_t nBytes = nSamples * descriptor.getNRxChannels() * sampleSize;

    ARRUS_REQUIRES_AT_MOST(outputAddress + nBytes, descriptor.getDdrSize(),
                           format("Total data size cannot exceed 4GiB (device {})", getDeviceId().toString()));
    ius4oem->ScheduleReceive(entryId, outputAddress, nSamplesRaw, sampleOffset + startSampleRaw,
                             op.getRxDecimationFactor() - 1, rxMapId, nullptr);
    return nBytes;
}

size_t Us4OEMImpl::scheduleReceiveRF(size_t outputAddress, uint32 startSample, uint32 endSample, uint16 entryId,
                                     const TxRxParameters &op, uint16 rxMapId) {
    const uint32 startSampleRaw = startSample * op.getRxDecimationFactor();
    const uint32 sampleOffset = ius4oem->GetTxOffset();
    const size_t nSamples = endSample - startSample;
    const size_t nSamplesRaw = nSamples;
    const size_t sampleSize = sizeof(RawDataType);
    const size_t nBytes = nSamples * descriptor.getNRxChannels() * sampleSize;
    ARRUS_REQUIRES_AT_MOST(outputAddress + nBytes, descriptor.getDdrSize(),
                           format("Total data size cannot exceed 4GiB (device {})", getDeviceId().toString()));
    ius4oem->ScheduleReceive(entryId, outputAddress, nSamplesRaw, sampleOffset + startSampleRaw,
                             op.getRxDecimationFactor() - 1, rxMapId, nullptr);
    return nBytes;
}

/**
 * Programs data acquisitions ("ScheduleReceive" part).
 *
 * 'element' here means the result data frame of the given operations sequence (times nRepeats)
 * This method programs us4OEM sequencer to fill the us4OEM memory with the acquired data
 * us4oem RXDMA output address.
*/
Us4OEMBuffer Us4OEMImpl::uploadAcquisition(const TxParametersSequenceColl &sequences, uint16 rxBufferSize,
                                           const std::optional<DigitalDownConversion> &ddc,
                                           const Us4OEMRxMappingRegister &rxMappingRegister) {
    bool isDDCOn = ddc.has_value();

    using BatchId = uint16;
    using SequenceId = uint16;
    using RepetitionId = uint16;
    using OpId = uint16;

    Us4OEMBufferBuilder builder;

    auto nSequences = ARRUS_SAFE_CAST(sequences.size(), SequenceId);
    size_t outputAddress = 0;
    size_t arrayStartAddress = 0;
    size_t elementStartAddress = 0;
    uint16 entryId = 0;
    for (BatchId batchId = 0; batchId < rxBufferSize; ++batchId) {
        // BUFFER ELEMENTS
        for (SequenceId seqId = 0; seqId < nSequences; ++seqId) {
            unsigned int totalSamples = 0;// Total number of samples in an array.
            // SEQUENCES
            Us4OEMBufferArrayParts parts;
            const auto &seq = sequences.at(seqId);
            for (RepetitionId repeatId = 0; repeatId < seq.getNRepeats(); ++repeatId) {
                // REPETITIONS
                for (OpId opId = 0; opId < seq.size(); ++opId, ++entryId) {
                    // OPS
                    auto const &op = seq.at(opId);

                    auto [startSample, endSample] = op.getRxSampleRange().asPair();
                    size_t nSamples = endSample - startSample;
                    auto rxMapId = rxMappingRegister.getMapId(seqId, opId);
                    size_t nBytes = 0;
                    if (isDDCOn) {
                        nBytes = scheduleReceiveDDC(outputAddress, startSample, endSample, entryId, op, rxMapId, ddc);
                    } else {
                        nBytes = scheduleReceiveRF(outputAddress, startSample, endSample, entryId, op, rxMapId);
                    }
                    if (batchId == 0) {
                        size_t partSize = 0;
                        if (!op.isRxNOP() || acceptRxNops) {
                            partSize = nBytes;
                        }
                        // Otherwise, make an empty part (i.e. partSize = 0).
                        // (note: the firing number will be needed for transfer configuration to release element in
                        // us4oem sequencer).
                        parts.emplace_back(outputAddress, partSize, seqId, entryId);
                    }
                    if (!op.isRxNOP() || acceptRxNops) {
                        // Also, allows rx nops for OEM that is acceptable, in order to acquire frame metadata.
                        // For example, the master module gathers frame metadata, so we cannot miss any of it.
                        // In all other cases, all RX nops are just overwritten.
                        outputAddress += nBytes;
                        totalSamples += static_cast<unsigned>(nSamples);
                    }
                }
            }
            framework::NdArray::Shape shape;
            if (isDDCOn) {
                shape = {totalSamples, 2, descriptor.getNRxChannels()};
            } else {
                shape = {totalSamples, descriptor.getNRxChannels()};
            }
            if (batchId == 0) {
                // Gather element layout.
                builder.add(Us4OEMBufferArrayDef{arrayStartAddress, framework::NdArrayDef{shape, DataType}, parts});
                arrayStartAddress = outputAddress;
            }
        }
        // entryId-1, because the firing should point to the last firing of this element
        ARRUS_REQUIRES_TRUE(entryId > 0, "Empty sequences are not supported");
        builder.add(
            Us4OEMBufferElement{elementStartAddress, outputAddress - elementStartAddress, (uint16) (entryId - 1)});
        elementStartAddress = outputAddress;
    }
    return builder.build();
}

void Us4OEMImpl::uploadTriggersIOBS(const TxParametersSequenceColl &sequences, uint16 rxBufferSize,
                                    Scheme::WorkMode workMode) {
    // Determine SRI values (last sequence PRI).
    std::vector<std::optional<float>> lastPriExtensions;
    for (const auto &sequence : sequences) {
        float totalPri = 0.0f;
        for (auto &op : sequence) {
            totalPri += op.getPri();
        }
        std::optional<float> lastPriExtension = std::nullopt;
        const auto &sri = sequence.getSri();
        if (sri.has_value()) {
            ARRUS_REQUIRES_TRUE_IAE(
                totalPri < sri.value(),
                format("Sequence repetition interval {} cannot be set, sequence total pri is equal {}", sri.value(),
                       totalPri));
            lastPriExtension = sri.value() - totalPri;
        }
        lastPriExtensions.push_back(lastPriExtension);
    }
    // Upload triggers and IOBS
    FiringId entryId = 0;
    auto nSequences = ARRUS_SAFE_CAST(sequences.size(), SequenceId);

    bool triggerSyncPerBatch = arrus::ops::us4r::Scheme::isWorkModeManual(workMode) || workMode == ops::us4r::Scheme::WorkMode::HOST;
    bool triggerSyncPerTxRx = workMode == ops::us4r::Scheme::WorkMode::MANUAL_OP;

    for (BatchId batchId = 0; batchId < rxBufferSize; ++batchId) {
        // BUFFER ELEMENTS
        for (SequenceId seqId = 0; seqId < nSequences; ++seqId) {
            // SEQUENCES
            const auto &seq = sequences.at(seqId);
            for (RepetitionId repeatId = 0; repeatId < seq.getNRepeats(); ++repeatId) {
                // REPETITIONS
                for (OpId opId = 0; opId < seq.size(); ++opId, ++entryId) {
                    // OPS
                    auto const &op = seq.at(opId);

                    bool isLastOp = opId == seq.size() - 1;
                    bool isLastRepeat = repeatId == seq.getNRepeats() - 1;
                    bool isLastSequence = seqId == sequences.size() - 1;
                    bool isCheckpoint = triggerSyncPerBatch && isLastOp && isLastRepeat && isLastSequence;
                    float pri = op.getPri();
                    if (isLastOp) {
                        auto lastPriExtension = lastPriExtensions.at(seqId);
                        if (lastPriExtension.has_value()) {
                            pri += lastPriExtension.value();
                        }
                    }
                    auto priMs = static_cast<unsigned int>(std::round(pri * 1e6));
                    ius4oem->SetTrigger(priMs, isCheckpoint || triggerSyncPerTxRx, entryId, isCheckpoint && externalTrigger,
                                        triggerSyncPerTxRx);
                    if (op.getBitstreamId().has_value() && isMaster()) {
                        ius4oem->SetFiringIOBS(entryId, bitstreamOffsets.at(op.getBitstreamId().value()));
                    }
                }
            }
        }
    }
}

void Us4OEMImpl::validate(const std::vector<TxRxParametersSequence> &sequences, uint16 rxBufferSize) {
    std::string deviceIdStr = getDeviceId().toString();
    for (size_t i = 0; i < sequences.size(); ++i) {
        const auto &seq = sequences.at(i);
        Us4OEMTxRxValidator seqValidator(format("{} tx rx sequence #{}", deviceIdStr, i),
                                         descriptor,
                                         static_cast<BitstreamId>(bitstreamOffsets.size()) );
        seqValidator.validate(seq);
        seqValidator.throwOnErrors();
    }
    // General sequence parameters.
    auto nFirings = getNumberOfFirings(sequences);
    auto nTriggers = getNumberOfTriggers(sequences, rxBufferSize);

    ARRUS_REQUIRES_AT_MOST(nFirings, 1024, format("Exceeded the maximum ({}) number of timeoutIds: {}", 1024, nFirings));
    const auto maxSequenceSize = descriptor.getTxRxSequenceLimits().getSize().end();
    ARRUS_REQUIRES_AT_MOST(nTriggers, maxSequenceSize,
                           format("Exceeded the maximum ({}) number of triggers: {}", maxSequenceSize, nTriggers));
}

float Us4OEMImpl::getTxRxTime(float rxTime) const {
    float txrxTime = 0.0f;
    if (reprogrammingMode == Us4OEMSettings::ReprogrammingMode::SEQUENTIAL) {
        txrxTime = rxTime + descriptor.getSequenceReprogrammingTime();
    } else if (reprogrammingMode == Us4OEMSettings::ReprogrammingMode::PARALLEL) {
        txrxTime = std::max(rxTime, descriptor.getSequenceReprogrammingTime());
    } else {
        throw IllegalArgumentException(
            format("Unrecognized reprogramming mode: {}", static_cast<size_t>(reprogrammingMode)));
    }
    return txrxTime;
}

Us4OEMRxMappingRegister Us4OEMImpl::setRxMappings(const TxParametersSequenceColl &sequences) {
    Us4OEMRxMappingRegisterBuilder builder{static_cast<FrameChannelMapping::Us4OEMNumber>(getDeviceId().getOrdinal()),
                                           acceptRxNops, channelMapping, descriptor.getNRxChannels()};
    builder.add(sequences);
    auto mappingRegister = builder.build();
    for (auto const &[mapId, map] : mappingRegister.getMappings()) {
        ius4oem->SetRxChannelMapping(map, mapId);
    }
    return mappingRegister;
}

float Us4OEMImpl::getSamplingFrequency() { return descriptor.getSamplingFrequency(); }

float Us4OEMImpl::getRxTime(const TxRxParameters &op, float samplingFrequency) {
    auto sampleRange = op.getRxSampleRange().asPair();
    float nSamples = static_cast<float>(std::get<1>(sampleRange));
    auto &pulse = op.getTxPulse();
    float txTime = pulse.getPulseLength();
    float rxTime = nSamples / samplingFrequency;
    // TODO consider txTime+rxTime
    rxTime = std::max(txTime, rxTime);
    return std::max(descriptor.getMinRxTime(), (float) rxTime + descriptor.getRxTimeEpsilon());
}

void Us4OEMImpl::start() { this->startTrigger(); }

void Us4OEMImpl::stop() { this->stopTrigger(); }

void Us4OEMImpl::syncTrigger() { this->ius4oem->TriggerSync(); }

void Us4OEMImpl::setTgcCurve(const std::vector<TxRxParametersSequence> &sequences) {
    // Make sure all TGC curve are the same.
    if (sequences.empty()) {
        return;
    }
    bool allCurvesTheSame = true;
    const auto &referenceCurve = sequences.at(0).getTgcCurve();
    for (size_t i = 1; i < sequences.size(); ++i) {
        const auto &s = sequences.at(i).getTgcCurve();
        if (s != referenceCurve) {
            allCurvesTheSame = false;
            break;
        }
    }
    ARRUS_REQUIRES_TRUE_IAE(allCurvesTheSame, "TGC curves for all sequences should be exactly the same.");
    setTgcCurve(sequences.at(0).getTgcCurve());
}

void Us4OEMImpl::setTgcCurve(const ops::us4r::TGCCurve &tgc) {
    RxSettingsBuilder rxSettingsBuilder(this->rxSettings);
    this->rxSettings = RxSettingsBuilder(this->rxSettings).setTgcSamples(tgc).build();
    setRxSettings(this->rxSettings);
}

Ius4OEMRawHandle Us4OEMImpl::getIUs4OEM() { return ius4oem.get(); }

void Us4OEMImpl::enableSequencer(bool resetSequencerPointer) {
    bool txConfOnTrigger = false;
    switch (reprogrammingMode) {
    case Us4OEMSettings::ReprogrammingMode::SEQUENTIAL: txConfOnTrigger = false; break;
    case Us4OEMSettings::ReprogrammingMode::PARALLEL: txConfOnTrigger = true; break;
    }
    this->ius4oem->EnableSequencer(txConfOnTrigger, resetSequencerPointer);
}

std::vector<uint8_t> Us4OEMImpl::getChannelMapping() { return channelMapping; }

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

uint32_t Us4OEMImpl::getTxOffset() { return ius4oem->GetTxOffset(); }

uint32_t Us4OEMImpl::getOemVersion() { return ius4oem->GetOemVersion(); }

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

void Us4OEMImpl::setAfeDemod(const std::optional<DigitalDownConversion> &ddc) {
    if (ddc.has_value()) {
        auto &value = ddc.value();
        setAfeDemod(value.getDemodulationFrequency(), value.getDecimationFactor(), value.getFirCoefficients().data(),
                    value.getFirCoefficients().size());
    } else {
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

const char *Us4OEMImpl::getSerialNumber() { return this->serialNumber.get().c_str(); }

const char *Us4OEMImpl::getRevision() { return this->revision.get().c_str(); }

BitstreamId Us4OEMImpl::addIOBitstream(const std::vector<uint8_t> &levels, const std::vector<uint16_t> &periods) {
    ARRUS_REQUIRES_EQUAL_IAE(levels.size(), periods.size());
    uint16 bitstreamOffset = 0;
    uint16 bitstreamId = 0;
    if (!bitstreamOffsets.empty()) {
        bitstreamId = int16(bitstreamOffsets.size());
        bitstreamOffset = bitstreamOffsets.at(bitstreamId - 1) + bitstreamSizes.at(bitstreamId - 1);
    }
    setIOBitstreamForOffset(bitstreamOffset, levels, periods);
    bitstreamOffsets.push_back(bitstreamOffset);
    bitstreamSizes.push_back(uint16(levels.size()));
    return bitstreamId;
}

void Us4OEMImpl::setIOBitstream(BitstreamId bitstreamId, const std::vector<uint8_t> &levels,
                                const std::vector<uint16_t> &periods) {
    ARRUS_REQUIRES_EQUAL_IAE(levels.size(), periods.size());
    ARRUS_REQUIRES_TRUE(bitstreamId < bitstreamOffsets.size(), "The bitstream with the given id does not exists.");
    if (bitstreamId != bitstreamOffsets.size() - 1) {
        ARRUS_REQUIRES_EQUAL_IAE(levels.size(), bitstreamSizes.at(bitstreamId));
    }
    // Allow to change the last bitstream size.
    setIOBitstreamForOffset(bitstreamOffsets.at(bitstreamId), levels, periods);
    bitstreamSizes[bitstreamId] = static_cast<uint16>(levels.size());
}

void Us4OEMImpl::setIOBitstreamForOffset(uint16 bitstreamOffset, const std::vector<uint8_t> &levels,
                                         const std::vector<uint16_t> &periods) {
    ARRUS_REQUIRES_EQUAL_IAE(levels.size(), periods.size());
    size_t nRegisters = static_cast<uint16_t>(levels.size());
    for (uint16_t i = 0; i < nRegisters; ++i) {
        ius4oem->SetIOBSRegister(bitstreamOffset + i, levels[i], i == (nRegisters - 1), periods[i]);
    }
}

size_t Us4OEMImpl::getNumberOfTriggers(const TxParametersSequenceColl &sequences, uint16 rxBufferSize) {
    return std::accumulate(std::begin(sequences), std::end(sequences), size_t(0),
                           [=](const auto &a, const auto &b) { return a + b.size() * b.getNRepeats() * rxBufferSize; });
}

size_t Us4OEMImpl::getNumberOfFirings(const std::vector<TxRxParametersSequence> &sequences) {
    return std::accumulate(std::begin(sequences), std::end(sequences), size_t(0),
                           [](const auto &a, const auto &b) { return a + b.size(); });
}

void Us4OEMImpl::setTxDelays(const std::vector<bool> &txAperture, const std::vector<float> &delays, uint16 firingId,
                             size_t delaysId, const std::unordered_set<ChannelIdx> &maskedChannelsTx) {
    ARRUS_REQUIRES_EQUAL_IAE(txAperture.size(), delays.size());
    for (uint8 ch = 0; ch < ARRUS_SAFE_CAST(txAperture.size(), uint8); ++ch) {
        bool bit = txAperture.at(ch);
        float delay = 0.0f;
        if (bit && !setContains(maskedChannelsTx, static_cast<ChannelIdx>(ch))) {
            delay = delays.at(ch);
        }
        ius4oem->SetTxDelay(ch, delay, firingId, delaysId);
    }
}

void Us4OEMImpl::clearCallbacks() {
    this->ius4oem->ClearCallbacks();
}

std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> Us4OEMImpl::filterAperture(
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> aperture,
    const std::unordered_set<ChannelIdx> &channelsMask) {
    for (auto channel:channelsMask) {
        aperture[channel] = false;
    }
    return aperture;
}

Us4OEMDescriptor Us4OEMImpl::getDescriptor() const {
    return descriptor;
}

void Us4OEMImpl::setMaximumPulseLength(std::optional<float> maxLength) {
    // 2 means OEM+
    // this is the only type of OEM that currently can have a maxLength != nullopt
    if(ius4oem->GetOemVersion() != 2 && maxLength.has_value()) {
        throw IllegalArgumentException("Currently it is possible to set maxLength value only for OEM+ (type 2)");
    }
    TxLimitsBuilder txBuilder{this->descriptor.getTxRxSequenceLimits().getTxRx().getTx()};
    if(maxLength.has_value()) {
        txBuilder.setPulseLength(Interval<float>{0.0f, maxLength.value()});
    }
    else {
        // Set the default setting.
        auto defaultLimits = Us4OEMDescriptorFactory::getDescriptor(ius4oem, isMaster()).getTxRxSequenceLimits().getTxRx().getTx().getPulseLength();
        txBuilder.setPulseLength(defaultLimits);
    }
    TxLimits txLimits = txBuilder.build();
    TxRxSequenceLimitsBuilder seqBuilder{descriptor.getTxRxSequenceLimits()};
    seqBuilder.setTxRxLimits(txLimits, descriptor.getTxRxSequenceLimits().getTxRx().getRx(),
                             descriptor.getTxRxSequenceLimits().getTxRx().getPri());
    Us4OEMDescriptorBuilder builder{descriptor};
    builder.setTxRxSequenceLimits(seqBuilder.build());
    // Set the new descriptor.
    descriptor = builder.build();
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

void Us4OEMImpl::waitForIrq(unsigned int irq, std::optional<long long> timeout) {
    this->irqEvents.at(irq).wait(timeout);
}

void Us4OEMImpl::sync(std::optional<long long> timeout) {
    logger->log(LogSeverity::TRACE, "Waiting for EVENTDONE IRQ");
    auto eventDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::EVENTDONE);
    this->waitForIrq(eventDoneIrq, timeout);
}

void Us4OEMImpl::setWaitForEventDone() {
    auto eventDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::EVENTDONE);
    irqEvents.at(eventDoneIrq).resetCounters();
    ius4oem->RegisterCallback(IUs4OEM::MSINumber::EVENTDONE, [eventDoneIrq, this]() {
        this->irqEvents.at(eventDoneIrq).notifyOne();
    });
}

void Us4OEMImpl::setWaitForHVPSMeasurementDone() {
    ius4oem->EnableHVPSMeasurementReadyIRQ();
    auto measurementDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE);
    irqEvents.at(measurementDoneIrq).resetCounters();
    ius4oem->RegisterCallback(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE, [measurementDoneIrq, this]() {
        this->irqEvents.at(measurementDoneIrq).notifyOne();
    });
}

void Us4OEMImpl::waitForHVPSMeasurementDone(std::optional<long long> timeout) {
    logger->log(LogSeverity::TRACE, "Waiting for HVPS Measurement done IRQ");
    auto measurementDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE);
    this->waitForIrq(measurementDoneIrq, timeout);
}

float Us4OEMImpl::getActualTxFrequency(float frequency) {
    return ius4oem->GetOCWSFrequency(frequency);
}
void Us4OEMImpl::setRxSettings(const RxSettings &settings) {
    ius4oem->SetRxSettings(settings, false);
}

}// namespace arrus::devices
