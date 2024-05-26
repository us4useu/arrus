#include "Us4OEMImpl.h"

#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

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
#include "arrus/core/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/external/ius4oem/ActiveTerminationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
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
    setRxSettingsPrivate(this->rxSettings, true);
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
    default: throw IllegalArgumentException(format("Unsupported HPF corner frequency: {}", frequency));
    }
    ius4oem->AfeEnableHPF();
    ius4oem->AfeSetHPFCornerFrequency(coefficient);
}

void Us4OEMImpl::disableHpf() { ius4oem->AfeDisableHPF(); }
Interval<Voltage> Us4OEMImpl::getAcceptedVoltageRange() { return Interval<Voltage>{0, 90}; }

void Us4OEMImpl::resetAfe() { ius4oem->AfeSoftReset(); }

Us4OEMUploadResult Us4OEMImpl::upload(const TxParametersSequenceColl &sequences, uint16 rxBufferSize,
                                      Scheme::WorkMode workMode, const std::optional<DigitalDownConversion> &ddc,
                                      const std::vector<arrus::framework::NdArray> &txDelays) {
    std::unique_lock<std::mutex> lock{stateMutex};
    validate(sequences, rxBufferSize);
    setTgcCurve(sequences);
    ius4oem->ResetSequencer();
    ius4oem->SetNumberOfFirings(getNumberOfFirings(sequences));
    ius4oem->ClearScheduledReceive();
    ius4oem->ResetCallbacks();
    auto rxMappingRegister = setRxMappings(sequences);
    this->isDecimationFactorAdjustmentLogged = false;
    uploadFirings(sequences, ddc, txDelays, rxMappingRegister);
    // For us4OEM+ the method below must be called right after programming TX/RX, and before calling ScheduleReceive.
    ius4oem->SetNTriggers(getNumberOfTriggers(sequences, rxBufferSize));
    auto bufferDef = uploadAcquisition(sequences, rxBufferSize, ddc, rxMappingRegister);
    uploadTriggersIOBS(sequences, rxBufferSize, workMode);
    setAfeDemod(ddc);
    return Us4OEMUploadResult{bufferDef, rxMappingRegister.acquireFCMs()};
}

void Us4OEMImpl::setTgcCurve(const ops::us4r::TGCCurve &tgc) {
    RxSettingsBuilder rxSettingsBuilder(this->rxSettings);
    this->rxSettings = RxSettingsBuilder(this->rxSettings).setTgcSamples(tgc)->build();
    setTgcCurve(this->rxSettings);
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
            auto [startSample, endSample] = op.getRxSampleRange().asPair();
            float decimationFactor = isDDCOn ? ddc->getDecimationFactor() : (float) op.getRxDecimationFactor();
            setCurrentSamplingFrequency(descriptor.getSamplingFrequency() / decimationFactor);
            float rxTime = getRxTime(endSample, this->currentSamplingFrequency);
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
            ius4oem->SetRxTime(rxTime, firingId);
            ius4oem->SetRxDelay(op.getRxDelay(), firingId);
        }
    }
    // Set the last profile as the current TX delay
    // (the last one is the one provided in the Sequence.ops.Tx.delays property).
    ius4oem->SetTxDelays(txDelays.size());
}

size_t Us4OEMImpl::scheduleReceiveDDC(size_t outputAddress, uint16 startSample, uint16 endSample, uint16 entryId,
                                      const TxRxParameters &op, uint16 rxMapId,
                                      const std::optional<DigitalDownConversion> &ddc) {
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

size_t Us4OEMImpl::scheduleReceiveRF(size_t outputAddress, uint16 startSample, uint16 endSample, uint16 entryId,
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
                    bool isTriggerSync = workMode == Scheme::WorkMode::HOST || workMode == Scheme::WorkMode::MANUAL;
                    bool isLastOp = opId == seq.size() - 1;
                    bool isLastRepeat = repeatId == seq.getNRepeats() - 1;
                    bool isLastSequence = seqId == sequences.size() - 1;
                    bool isCheckpoint = isTriggerSync && isLastOp && isLastRepeat && isLastSequence;
                    float pri = op.getPri();
                    if (isLastOp) {
                        auto lastPriExtension = lastPriExtensions.at(seqId);
                        if (lastPriExtension.has_value()) {
                            pri += lastPriExtension.value();
                        }
                    }
                    auto priMs = static_cast<unsigned int>(std::round(pri * 1e6));
                    ius4oem->SetTrigger(priMs, isCheckpoint, entryId, isCheckpoint && externalTrigger);
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

    ARRUS_REQUIRES_AT_MOST(nFirings, 1024, format("Exceeded the maximum ({}) number of firings: {}", 1024, nFirings));
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

float Us4OEMImpl::getRxTime(size_t nSamples, float samplingFrequency) {
    return std::max(descriptor.getMinRxTime(), (float) nSamples / samplingFrequency + descriptor.getRxTimeEpsilon());
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

// AFE setters
void Us4OEMImpl::setTgcCurve(const RxSettings &afeCfg) {
    const ops::us4r::TGCCurve &tgc = afeCfg.getTgcSamples();
    bool applyCharacteristic = afeCfg.isApplyTgcCharacteristic();

    auto tgcMax = static_cast<float>(afeCfg.getPgaGain() + afeCfg.getLnaGain());
    auto tgcMin = tgcMax - RxSettings::TGC_ATTENUATION_RANGE;
    // Set.
    if (tgc.empty()) {
        ius4oem->TGCDisable();
    } else {
        std::vector<float> actualTgc = tgc;
        // Normalize to [0, 1].
        for (auto &val : actualTgc) {
            val = (val - tgcMin) / RxSettings::TGC_ATTENUATION_RANGE;
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
    return std::accumulate(std::begin(sequences), std::end(sequences), 0,
                           [=](const auto &a, const auto &b) { return a + b.size() * b.getNRepeats() * rxBufferSize; });
}

size_t Us4OEMImpl::getNumberOfFirings(const std::vector<TxRxParametersSequence> &sequences) {
    return std::accumulate(std::begin(sequences), std::end(sequences), 0,
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

}// namespace arrus::devices
