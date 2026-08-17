#include "Us4OEMImpl.h"

#include <chrono>
#include <cmath>
#include <format>
#include <thread>
#include <utility>
#include <regex>
#include <std4us/string.h>

#include "Us4OEMDescriptorFactory.h"
#include "Us4OEMTxRxValidator.h"
#include "arrus/common/asserts.h"
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
#include "utils.h"
#include "arrus/core/devices/us4r/TxWaveformConverter.h"

namespace arrus::devices {
// TODO migrate this source to us4r subspace

using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem, std::vector<uint8_t> channelMapping, RxSettings rxSettings,
                       Us4OEMSettings::ReprogrammingMode reprogrammingMode, Us4OEMDescriptor descriptor,
                       bool acceptRxNops = false)
    : Us4OEMImplBase(id), logger{getLoggerFactory()->getLogger()}, ius4oem(std::move(ius4oem)),
      descriptor(std::move(descriptor)),
      channelMapping(std::move(channelMapping)), reprogrammingMode(reprogrammingMode),
      rxSettings(std::move(rxSettings)),
      serialNumber([this]() { return this->ius4oem->getSerialNumber(); }),
      revision([this]() { return this->ius4oem->getRevisionNumber(); }), acceptRxNops(acceptRxNops) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    setTestPattern(RxTestPattern::OFF);
    disableAfeDemod();
    this->ius4oem->setRxSettings(this->rxSettings, true);
    setCurrentSamplingFrequency(this->descriptor.getSamplingFrequency());
}

Us4OEMImpl::~Us4OEMImpl() {
    try {
        logger->debug("Destroying handle");
    } catch (const std::exception &e) {
        std::cerr << std::format("Exception while calling us4oem destructor: {}", e.what()) << std::endl;
    }
    logger->debug("Us4OEM handle destroyed.");
}

bool Us4OEMImpl::isMaster() { return descriptor.isMaster(); }

void Us4OEMImpl::startTrigger() {
    if (isMaster()) {
        ius4oem->triggerStart();
    }
}

void Us4OEMImpl::stopTrigger() {
    if (isMaster()) {
        ius4oem->triggerStop();
    }
}

uint16_t Us4OEMImpl::getAfe(uint8_t address) { return ius4oem->afeReadRegister(0, address); }

void Us4OEMImpl::setAfe(uint8_t address, uint16_t value) {
    ius4oem->afeWriteRegister(0, address, value);
    ius4oem->afeWriteRegister(1, address, value);
}

void Us4OEMImpl::enableAfeDemod() { ius4oem->afeDemodEnable(); }

void Us4OEMImpl::setAfeDemodConfig(uint8_t decInt, uint8_t decQuarters, const float *firCoeffs, uint16_t firLength,
                                   float freq, float gain) {
    const auto availableGains = DDC_GAIN_MAP.getAvailableValues();
    ARRUS_REQUIRES_TRUE_IAE(setContains(availableGains, gain),
                            std::format("Digital Down Conversion gain should be one of: {}",
                               std4us::join(availableGains, ", ")));
    auto actualValue = DDC_GAIN_MAP.get(gain);
    ius4oem->afeDemodConfig(decInt, decQuarters, firCoeffs, firLength, freq, actualValue);
}

void Us4OEMImpl::setAfeDemodDefault() { ius4oem->afeDemodSetDefault(); }

void Us4OEMImpl::setAfeDemodDecimationFactor(uint8_t integer) { ius4oem->afeDemodSetDecimationFactor(integer); }

void Us4OEMImpl::setAfeDemodDecimationFactor(uint8_t integer, uint8_t quarters) {
    ius4oem->afeDemodSetDecimationFactorQuarters(integer, quarters);
}

void Us4OEMImpl::setAfeDemodFrequency(float frequency) {
    // Note: us4r-api expects frequency in Hz.
    ius4oem->afeDemodSetDemodFrequency(frequency / 1e6f);
    ius4oem->afeDemodFsweepDisable();
}

void Us4OEMImpl::setAfeDemodFrequency(float startFrequency, float stopFrequency) {
    // Note: us4r-api expects frequency in Hz.
    ius4oem->afeDemodSetDemodFrequency(startFrequency / 1e6f, stopFrequency / 1e6f);
    ius4oem->afeDemodFsweepEnable();
}

float Us4OEMImpl::getAfeDemodStartFrequency() { return ius4oem->afeDemodGetStartFrequency(); }

float Us4OEMImpl::getAfeDemodStopFrequency() { return ius4oem->afeDemodGetStopFrequency(); }

void Us4OEMImpl::setAfeDemodFsweepROI(uint16_t startSample, uint16_t stopSample) {
    ius4oem->afeDemodSetFsweepRoi(startSample, stopSample);
}

void Us4OEMImpl::writeAfeFIRCoeffs(const int16_t *coeffs, uint16_t length) {
    ius4oem->afeDemodWriteFirCoeffs(coeffs, length);
}

void Us4OEMImpl::writeAfeFIRCoeffs(const float *coeffs, uint16_t length) {
    ius4oem->afeDemodWriteFirCoeffs(coeffs, length);
}

void Us4OEMImpl::setLnaHpfCornerFrequency(uint32_t frequency) {
    ius4oem->afeSetLnaHpfCornerFrequency(frequency);
}

void Us4OEMImpl::disableLnaHpf() { ius4oem->afeDisableLnaHpf(); }

void Us4OEMImpl::setAdcHpfCornerFrequency(uint32_t frequency) {
    ius4oem->afeSetAdcHpfCornerFrequency(frequency);
}

void Us4OEMImpl::disableAdcHpf() { ius4oem->afeDisableAdcHpf(); }

Interval<Voltage> Us4OEMImpl::getAcceptedVoltageRange() { return Interval<Voltage>{0, 90}; }

void Us4OEMImpl::resetAfe() { ius4oem->afeSoftReset(); }

Us4OEMUploadResult Us4OEMImpl::upload(const std::vector<us4r::TxRxParametersSequence> &sequences,
                                      uint16 rxBufferSize, ops::us4r::Scheme::WorkMode workMode,
                                      const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                                      const std::vector<std::vector<arrus::framework::NdArray>> &txDelays,
                                      const std::vector<TxTimeout> &txTimeouts) {
    std::unique_lock<std::mutex> lock{stateMutex};
    validate(sequences, rxBufferSize);
    setTgcCurve(sequences);
    ius4oem->resetSequencer();
    ius4oem->setNumberOfFirings(ARRUS_SAFE_CAST(getNumberOfFirings(sequences), uint16_t));
    ius4oem->clearScheduledReceive();
    ius4oem->resetRuntimeCallbacks();
    auto rxMappingRegister = setRxMappings(sequences);
    this->isDecimationFactorAdjustmentLogged = false;
    setTxTimeouts(txTimeouts);
    uploadFirings(sequences, ddc, txDelays, rxMappingRegister);
    // For us4OEM+ the method below must be called right after programming TX/RX, and before calling scheduleReceive.
    ius4oem->setNTriggers(ARRUS_SAFE_CAST(getNumberOfTriggers(sequences, rxBufferSize), uint16_t));
    auto [bufferDef, rxTimeOffset] = uploadAcquisition(sequences, rxBufferSize, ddc, rxMappingRegister);
    uploadTriggersIOBS(sequences, rxBufferSize, workMode);
    setAfeDemod(ddc);
    if(Scheme::isWorkModeManual(workMode)) {
        setWaitForEventDone();
    }
    return Us4OEMUploadResult{bufferDef, rxMappingRegister.acquireFCMs(), rxTimeOffset};
}
void Us4OEMImpl::setTxTimeouts(const std::vector<TxTimeout> &txTimeouts) {
    if(!txTimeouts.empty()) {
        ius4oem->enableTxTimeout();
        for(size_t n = 0; n < txTimeouts.size(); ++n) {
            ius4oem->setTxTimeout((uint8_t)n, txTimeouts[n]);
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
                               const std::vector<std::vector<arrus::framework::NdArray>> &txDelays,
                               const Us4OEMRxMappingRegister &rxMappingRegister) {
    using SequenceId = uint16;
    using OpId = uint16;

    bool isDDCOn = ddc.has_value();
    const Us4OEMChannelsGroupsMask emptyChannelGroups;

    // Reset the currently selected profiles.
    currentTxDelayProfileIds = std::vector<size_t>(sequences.size());
    // us4OEM sequencer firing/entry id (global).
    OpId firingId = 0;
    for (SequenceId sequenceId = 0; sequenceId < ARRUS_SAFE_CAST(sequences.size(), SequenceId); ++sequenceId) {
        auto const &sequence = sequences[sequenceId];
        for (OpId opId = 0; opId < ARRUS_SAFE_CAST(sequence.size(), OpId); ++opId, ++firingId) {
            auto const &op = sequence.at(opId);
            logger->trace("Setting sequence {}, TX/RX {}: NOP? {}, definition: {}", sequenceId, opId, op.isNOP(),
                          std4us::to_string(op));
            // TX
            auto txAperture = arrus::toBitset<Us4OEMDescriptor::N_TX_CHANNELS>(op.getTxAperture());
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
                                    std::format("Total time required for a single TX/RX ({}) should not exceed PRI ({})",
                                           txrxTime, op.getPri()));
            // Upload
            ius4oem->setActiveChannelGroup(channelsGroups, firingId);
            ius4oem->setTxAperture(filteredTxAperture, firingId);
            ius4oem->setRxAperture(filteredRxAperture, firingId);
            ius4oem->setRxDelay(op.getRxDelay(), firingId);
            // Delays
            size_t nProfiles = 0;
            if(!txDelays.empty()) {
                // Set delay definition tables, specific for the given sequence.
                const auto &sequenceDelays = txDelays.at(sequenceId);
                nProfiles = sequenceDelays.size();
                for (size_t delaysId = 0; delaysId < sequenceDelays.size(); ++delaysId) {
                    auto delays = sequenceDelays.at(delaysId).row(opId).toVector<float>();

                    setTxDelays(op.getTxAperture(), delays, firingId, delaysId, op.getMaskedChannelsTx(), sequenceId);
                }
            }
            // Then set the profile from the input sequence (for backward-compatibility).
            // NOTE: this might look redundant and it is, however it simplifies the changes for v0.9.0 a lot
            // and reduces the risk of causing new bugs in the whole mapping implementation.
            setTxDelays(op.getTxAperture(), op.getTxDelays(), firingId, nProfiles, op.getMaskedChannelsTx(),sequenceId);
            // Remember what is the currently selected TX delay profile.
            currentTxDelayProfileIds.at(sequenceId) = nProfiles;
            if(isOEMPlus()) {
                auto waveform = op.getTxWaveform();
                ius4oem->setCustomSequenceWaveform(firingId, TxWaveformConverter::toPulser(waveform));
            }
            else {
                // Legacy OEM.
                auto pulse = Pulse::fromWaveform(op.getTxWaveform());
                ARRUS_REQUIRES_TRUE(
                    pulse.has_value(),
                    std::format("Couldn't get the correct TX pulse for the waveform declared in the firing {}", firingId));
                ius4oem->setTxFreqency(pulse.value().getCenterFrequency(), firingId);
                auto nTxHalfPeriods = static_cast<uint32>(pulse.value().getNPeriods()*2);
                ius4oem->setTxHalfPeriods(nTxHalfPeriods, firingId);
                ius4oem->setTxInvert(pulse.value().isInverse(), firingId);
            }
            ius4oem->setRxTime(rxTime, firingId);
            if(isOEMPlus() && op.getTxTimeoutId().has_value()) {
                ius4oem->setFiringTxTimoutId(firingId, op.getTxTimeoutId().value());
            }
        }
    }
    // Set the last profile as the current TX delay
    // (the last one is the one provided in the Sequence.ops.Tx.delays property).
    ius4oem->setTxDelays(currentTxDelayProfileIds);
}

std::pair<size_t, float> Us4OEMImpl::scheduleReceiveDDC(size_t outputAddress,
                                                        uint32 startSample, uint32 endSample, uint16 entryId,
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
        this->logger->warn("Decimation factor {} requires start offset to be multiple "
                                          "of {}. Offset adjusted to {}.",
                                          ddc->getDecimationFactor(), div, startSample);
    }
    // Start sample, after transforming to the system number of cycles.
    // The start sample should be provided to the us4r-api
    // as for the nominal sampling frequency of us4OEM, i.e. 65 MHz.
    const uint32 startSampleRaw = startSample * (uint32_t) ddc->getDecimationFactor();
    // Sample RX offset closest (and <= if possible) to the moment tx delay = 0,
    // and sample RX offset time residue (time difference between sampleRxOffset and the moment tx delay = 0).
    const auto [sampleRxOffset, sampleRxOffsetTimeResidue] = getTxStartSampleNumberAfeDemod(ddc->getDecimationFactor());
    // Number of samples to acquire per channel.
    const size_t nSamples = endSample - startSample;
    // Number of samples to be set on IUs4OEM::scheduleReceive
    const size_t nSamplesRaw = nSamples * 2;
    // Number of bytes a single sample takes (e.g. RF: a single int16, IQ: a pair of int16)
    const size_t sampleSize = 2 * sizeof(RawDataType);
    const size_t nBytes = nSamples * descriptor.getNRxChannels() * sampleSize;

    ARRUS_REQUIRES_AT_MOST(outputAddress + nBytes, descriptor.getDdrSize(),
                           std::format("Total data size cannot exceed 4GiB (device {})", getDeviceId().toString()));
    US4US_US4R_PROGRAMMING_CHUNK_PAUSE(entryId);
    ius4oem->scheduleReceive(entryId, outputAddress, nSamplesRaw, sampleRxOffset + startSampleRaw,
                             op.getRxDecimationFactor() - 1, rxMapId, nullptr);

    return std::make_pair(nBytes, sampleRxOffsetTimeResidue);
}

size_t Us4OEMImpl::scheduleReceiveRF(size_t outputAddress, uint32 startSample, uint32 endSample, uint16 entryId,
                                     const TxRxParameters &op, uint16 rxMapId) {
    const uint32 startSampleRaw = startSample * op.getRxDecimationFactor();
    const uint32 sampleRxOffset = descriptor.getSampleTxStart();
    const size_t nSamples = endSample - startSample;
    const size_t nSamplesRaw = nSamples;
    const size_t sampleSize = sizeof(RawDataType);
    const size_t nBytes = nSamples * descriptor.getNRxChannels() * sampleSize;
    ARRUS_REQUIRES_AT_MOST(outputAddress + nBytes, descriptor.getDdrSize(),
                           std::format("Total data size cannot exceed 4GiB (device {})", getDeviceId().toString()));
    US4US_US4R_PROGRAMMING_CHUNK_PAUSE(entryId);
    ius4oem->scheduleReceive(entryId, outputAddress, nSamplesRaw, sampleRxOffset + startSampleRaw,
                             op.getRxDecimationFactor() - 1, rxMapId, nullptr);
    return nBytes;
}

/**
 * Programs data acquisitions ("scheduleReceive" part).
 *
 * 'element' here means the result data frame of the given operations sequence (times nRepeats)
 * This method programs us4OEM sequencer to fill the us4OEM memory with the acquired data
 * us4oem RXDMA output address.
*/
std::pair<Us4OEMBuffer, float> Us4OEMImpl::uploadAcquisition(const TxParametersSequenceColl &sequences, uint16 rxBufferSize,
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
    float rxTimeOffset = 0; // actually, rxOffsetTimeResidue, but rxOffset (sampleRxOffset) is already compensated in scheduleReceiveDDC,
                            // so the rxOffsetTimeResidue is the only remaining offset. Now it's named rxTimeOffset for simplicity.
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
                        auto res = scheduleReceiveDDC(outputAddress, startSample, endSample, entryId, op, rxMapId, ddc);
                        nBytes = res.first;
                        rxTimeOffset = res.second;
                    } else {
                        nBytes = scheduleReceiveRF(outputAddress, startSample, endSample, entryId, op, rxMapId);
                    }
                    if (batchId == 0) {
                        size_t partSize = 0;
                        size_t actualNumberOfSamples = 0;
                        if (!op.isRxNOP() || acceptRxNops) {
                            partSize = nBytes;
                            actualNumberOfSamples = nSamples;
                        }
                        // Otherwise, make an empty part (i.e. partSize = 0).
                        // (note: the firing number will be needed for transfer configuration to release element in
                        // us4oem sequencer).
                        parts.emplace_back(outputAddress, partSize, seqId, entryId, ARRUS_SAFE_CAST(actualNumberOfSamples, uint32_t));
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
    return std::make_pair(builder.build(), rxTimeOffset);
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
                std::format("Sequence repetition interval {} cannot be set, sequence total pri is equal {}", sri.value(),
                       totalPri));
            lastPriExtension = sri.value() - totalPri;
        }
        lastPriExtensions.push_back(lastPriExtension);
    }
    // Upload triggers and IOBS
    FiringId entryId = 0;
    auto nSequences = ARRUS_SAFE_CAST(sequences.size(), SequenceId);

    bool triggerSyncPerBatch = isWaitForSoftMode(workMode);
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
                    // syncReq (interrupt: 3) only when we are hitting the last TX/RX in the sequence,
                    //  or we have the MANUAL_OP work mode (stop after each TX/RX)
                    // syncMode (external trigger): only when we are hitting the last TX/Rx in the sequence,
                    //  and the user configured the system to use "external trigger source"
                    // irqDone (interrupt: 4): only when we have the MANUAL_OP (signal the IRQ after each TX/RX)
                    //  or we have the MANUAL work mode, and we are hitting the last TX/RX in the TX/RX
                    //  (the IRQ = 4 is used to implement the synchronous version of the Us4r::trigger(sync=true))
                    US4US_US4R_PROGRAMMING_CHUNK_PAUSE(entryId);
                    ius4oem->setTrigger(priMs, isCheckpoint || triggerSyncPerTxRx, entryId, isCheckpoint && useSequenceTriggerCapability,
                                        triggerSyncPerTxRx || (isCheckpoint && workMode == ops::us4r::Scheme::WorkMode::MANUAL));
                    if (op.getBitstreamId().has_value() && isMaster()) {
                        ius4oem->setFiringIobs(entryId, bitstreamOffsets.at(op.getBitstreamId().value()));
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
        Us4OEMTxRxValidator seqValidator(std::format("{} tx rx sequence #{}", deviceIdStr, i),
                                         descriptor,
                                         static_cast<BitstreamId>(bitstreamOffsets.size()) );
        seqValidator.validate(seq);
        seqValidator.throwOnErrors();
    }
    // General sequence parameters.
    auto nFirings = getNumberOfFirings(sequences);
    auto nTriggers = getNumberOfTriggers(sequences, rxBufferSize);

    auto maxFirings = descriptor.getTxRxSequenceLimits().getMaxNumberOfFirings();

    ARRUS_REQUIRES_AT_MOST(nFirings, maxFirings, std::format("Exceeded the maximum ({}) number of firings: {}", maxFirings, nFirings));
    const auto maxSequenceSize = descriptor.getTxRxSequenceLimits().getSize().end();
    ARRUS_REQUIRES_AT_MOST(nTriggers, maxSequenceSize,
                           std::format("Exceeded the maximum ({}) number of triggers: {}", maxSequenceSize, nTriggers));
}

float Us4OEMImpl::getTxRxTime(float rxTime) const {
    float txrxTime = 0.0f;
    if (reprogrammingMode == Us4OEMSettings::ReprogrammingMode::SEQUENTIAL) {
        txrxTime = rxTime + descriptor.getSequenceReprogrammingTime();
    } else if (reprogrammingMode == Us4OEMSettings::ReprogrammingMode::PARALLEL) {
        txrxTime = std::max(rxTime, descriptor.getSequenceReprogrammingTime());
    } else {
        throw IllegalArgumentException(
            "Unrecognized reprogramming mode: {}", static_cast<size_t>(reprogrammingMode));
    }
    return txrxTime;
}

Us4OEMRxMappingRegister Us4OEMImpl::setRxMappings(const TxParametersSequenceColl &sequences) {
    Us4OEMRxMappingRegisterBuilder builder{static_cast<FrameChannelMapping::Us4OEMNumber>(getDeviceId().getOrdinal()),
                                           acceptRxNops, channelMapping, descriptor.getNRxChannels()};
    builder.add(sequences);
    auto mappingRegister = builder.build();
    for (auto const &[mapId, map] : mappingRegister.getMappings()) {
        ius4oem->setRxChannelMapping(map, mapId);
    }
    return mappingRegister;
}

float Us4OEMImpl::getSamplingFrequency() { return descriptor.getSamplingFrequency(); }

float Us4OEMImpl::getRxTime(const TxRxParameters &op, float samplingFrequency) {
    auto sampleRange = op.getRxSampleRange().asPair();
    float nSamples = static_cast<float>(std::get<1>(sampleRange));
    auto &waveform = op.getTxWaveform();
    float txTime = waveform.getTotalDuration();
    float rxTime = nSamples / samplingFrequency;
    // TODO consider txTime+rxTime
    rxTime = std::max(txTime, rxTime);
    return std::max(descriptor.getMinRxTime(), (float) rxTime + descriptor.getRxTimeEpsilon());
}

void Us4OEMImpl::start() { this->startTrigger(); }

void Us4OEMImpl::stop() { this->stopTrigger(); }

void Us4OEMImpl::syncTrigger() { this->ius4oem->triggerSync(); }

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

void Us4OEMImpl::enableSequencer(uint16 startEntry, bool dvddMask) {
    bool txConfOnTrigger = false;
    switch (reprogrammingMode) {
    case Us4OEMSettings::ReprogrammingMode::SEQUENTIAL: txConfOnTrigger = false; break;
    case Us4OEMSettings::ReprogrammingMode::PARALLEL: txConfOnTrigger = true; break;
    }
    this->ius4oem->enableSequencer(txConfOnTrigger, startEntry, dvddMask);
}

std::vector<uint8_t> Us4OEMImpl::getChannelMapping() { return channelMapping; }

float Us4OEMImpl::getFPGATemperature() { return ius4oem->getFpgaTemp(); }

float Us4OEMImpl::getUCDTemperature() { return ius4oem->getUcdTemp(); }

float Us4OEMImpl::getUCDExternalTemperature() { return ius4oem->getUcdExtTemp(); }

float Us4OEMImpl::getUCDMeasuredVoltage(uint8_t rail) { return ius4oem->getUcdVout(rail); }

float Us4OEMImpl::getMeasuredHVPVoltage() { return ius4oem->getMeasuredHvpVoltage(); }
float Us4OEMImpl::getMeasuredHVMVoltage() { return ius4oem->getMeasuredHvmVoltage(); }

void Us4OEMImpl::checkFirmwareVersion() {
    try {
        ius4oem->checkFirmwareVersion();
    } catch (const std::runtime_error &e) { throw arrus::IllegalStateException(e.what()); } catch (...) {
        throw arrus::IllegalStateException("Unknown exception while check firmware version.");
    }
}

uint32 Us4OEMImpl::getFirmwareVersion() { return ius4oem->getFirmwareVersion(); }

uint32 Us4OEMImpl::getTxFirmwareVersion() { return ius4oem->getTxFirmwareVersion(); }

uint32_t Us4OEMImpl::getOemVersion() { return ius4oem->getOemVersion(); }

void Us4OEMImpl::checkState() { this->checkFirmwareVersion(); }

void Us4OEMImpl::setTestPattern(RxTestPattern pattern) {
    switch (pattern) {
    case RxTestPattern::RAMP: ius4oem->enableTestPatterns(); break;
    case RxTestPattern::OFF: ius4oem->disableTestPatterns(); break;
    default: throw IllegalArgumentException("Unrecognized test pattern");
    }
}

std::pair<uint32_t, float> Us4OEMImpl::getTxStartSampleNumberAfeDemod(float ddcDecimationFactor) {
    //DDC RX offset (valid data offset)
    uint32_t txOffset = descriptor.getSampleTxStart();
    uint32_t rxOffset = 34u + (uint32_t)(16 * ddcDecimationFactor);
    uint32_t filterDelay = (uint32_t)(8 * ddcDecimationFactor);

    float decInt = 0;
    float decFloat = modf(ddcDecimationFactor, &decInt);

    uint32_t dataStep = (uint32_t)decInt;
    if (decFloat == 0.5f) {
        dataStep = (uint32_t)(2.0f * ddcDecimationFactor);
    } else if (decFloat == 0.25f || decFloat == 0.75f) {
        dataStep = (uint32_t)(4.0f * ddcDecimationFactor);
    }

    //Check if RX offset is higher than TX offset + filter delay
    if (rxOffset > txOffset + filterDelay) {
        //If so, do not adjust RX offset and log warning
        if(!this->isDecimationFactorAdjustmentLogged) {
            this->logger->info("Decimation factor {} causes RX data to start after the moment TX starts."
                                          " Delay TX by {} microseconds to align start of RX data with start of TX.",
                                          ddcDecimationFactor, (float)(rxOffset - txOffset - filterDelay)/65.0f);
            this->isDecimationFactorAdjustmentLogged = true;
        }
    } else {
        //Calculate RX offset pointing to DDC sample closest but lower than TX offset + filter delay
        rxOffset += ((txOffset + filterDelay - rxOffset) / dataStep) * dataStep;
    }

    float rxOffsetResidue = (float)(txOffset + filterDelay) - (float)(rxOffset);
    float rxOffsetTimeResidue = rxOffsetResidue / descriptor.getSamplingFrequency();

    return std::make_pair(rxOffset, rxOffsetTimeResidue);
}

float Us4OEMImpl::getCurrentSamplingFrequency() const {
    std::unique_lock<std::mutex> lock{stateMutex};
    return currentSamplingFrequency;
}

uint64_t Us4OEMImpl::getFPGAWallclock() { return ius4oem->getFpgaWallclock(); }

void Us4OEMImpl::setAfeDemod(const std::optional<DigitalDownConversion> &ddc) {
    if (ddc.has_value()) {
        auto &value = ddc.value();
        setAfeDemod(value.getDemodulationFrequency(), value.getDecimationFactor(), value.getFirCoefficients().data(),
                    value.getFirCoefficients().size(), value.getGain());
    } else {
        disableAfeDemod();
    }
}

void Us4OEMImpl::setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                             size_t nCoefficients, float gain) {
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
        throw IllegalArgumentException("Incorrect number of DDC FIR filter coefficients, should be {}, "
                                      "actual: {}",
                                      expectedNumberOfCoeffs, nCoefficients);
    }
    enableAfeDemod();
    setAfeDemodConfig(static_cast<uint8_t>(decInt), static_cast<uint8_t>(nQuarters), firCoefficients,
                      static_cast<uint16_t>(nCoefficients), demodulationFrequency, gain);
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
        ius4oem->setIobsRegister(bitstreamOffset + i, levels[i], i == (nRegisters - 1), periods[i]);
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
                             size_t delaysId, const std::unordered_set<ChannelIdx> &maskedChannelsTx,
                             SequenceId sequenceId) {
    ARRUS_REQUIRES_EQUAL_IAE(txAperture.size(), delays.size());
    std::vector<float> delaysToBeApplied(txAperture.size());
    for (uint8 ch = 0; ch < ARRUS_SAFE_CAST(txAperture.size(), uint8); ++ch) {
        bool bit = txAperture.at(ch);
        float delay = 0.0f;
        if (bit && !setContains(maskedChannelsTx, static_cast<ChannelIdx>(ch))) {
            delay = delays.at(ch);
        }
        delaysToBeApplied.at(ch) = delay;
    }
    ius4oem->setTxDelays(delaysToBeApplied, firingId, delaysId, ARRUS_SAFE_CAST(sequenceId, size_t));
}

void Us4OEMImpl::clearDMACallbacks() {
    this->ius4oem->resetDmaCallbacks();
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
    if(ius4oem->getOemVersion() != 2 && maxLength.has_value()) {
        throw IllegalArgumentException("Currently it is possible to set maxLength value only for OEM+ (type 2)");
    }
    TxLimitsBuilder txBuilder{this->descriptor.getTxRxSequenceLimits().getTxRx().getTx2()};
    if(maxLength.has_value()) {
        txBuilder.setPulseLength(Interval<float>{0.0f, maxLength.value()});
    }
    else {
        txBuilder.setPulseLength(Interval<float>{0.0f, 0.0f});
        // Set the default setting.
        auto defaultLimits = Us4OEMDescriptorFactory::getDescriptor(ius4oem, isMaster())
                                 .getTxRxSequenceLimits()
                                 .getTxRx()
                                 .getTx2().getPulseCycles();
        txBuilder.setPulseCycles(defaultLimits);
    }
    TxLimits txLimits = txBuilder.build();
    TxRxSequenceLimitsBuilder seqBuilder{descriptor.getTxRxSequenceLimits()};
    seqBuilder.setTxRxLimits(
        // TX amplitude level 1 / HV rail 1
        descriptor.getTxRxSequenceLimits().getTxRx().getTx1(),
        // TX amplitude level 2 / HV rail 0
        txLimits,
        descriptor.getTxRxSequenceLimits().getTxRx().getRx(),
        descriptor.getTxRxSequenceLimits().getTxRx().getPri());
    Us4OEMDescriptorBuilder builder{descriptor};
    builder.setTxRxSequenceLimits(seqBuilder.build());
    // Set the new descriptor.
    descriptor = builder.build();
}

HVPSMeasurement Us4OEMImpl::getHVPSMeasurement() {
    auto m = ius4oem->getHvpsMeasurements();
    HVPSMeasurementBuilder builder;
    builder.set(0, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::VOLTAGE, m.hvp0Voltage);
    builder.set(0, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::CURRENT, m.hvp0Current);
    builder.set(1, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::VOLTAGE, m.hvp1Voltage);
    builder.set(1, HVPSMeasurement::Polarity::PLUS, HVPSMeasurement::Unit::CURRENT, m.hvp1Current);
    builder.set(0, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::VOLTAGE, m.hvm0Voltage);
    builder.set(0, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::CURRENT, m.hvm0Current);
    builder.set(1, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::VOLTAGE, m.hvm1Voltage);
    builder.set(1, HVPSMeasurement::Polarity::MINUS, HVPSMeasurement::Unit::CURRENT, m.hvm1Current);
    return builder.build();
}

float Us4OEMImpl::setHVPSSyncMeasurement(uint16_t nSamples, float frequency) {
    return ius4oem->setHvpsSyncMeasurement(nSamples, frequency);
}

void Us4OEMImpl::waitForIrq(unsigned int irq, std::optional<long long> timeout) {
    this->irqEvents.at(irq).wait(timeout);
}

void Us4OEMImpl::sync(std::optional<long long> timeout) {
    logger->trace("Waiting for EVENTDONE IRQ");
    auto eventDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::EVENTDONE);
    this->waitForIrq(eventDoneIrq, timeout);
}

void Us4OEMImpl::setWaitForEventDone() {
    auto eventDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::EVENTDONE);
    irqEvents.at(eventDoneIrq).resetCounters();
    ius4oem->registerCallback(IUs4OEM::MSINumber::EVENTDONE, [eventDoneIrq, this]() {
        this->irqEvents.at(eventDoneIrq).notifyOne();
    });
}

void Us4OEMImpl::setWaitForHVPSMeasurementDone() {
    ius4oem->enableHvpsMeasurementReadyIrq();
    auto measurementDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE);
    irqEvents.at(measurementDoneIrq).resetCounters();
    ius4oem->registerCallback(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE, [measurementDoneIrq, this]() {
        this->irqEvents.at(measurementDoneIrq).notifyOne();
    });
}

void Us4OEMImpl::waitForHVPSMeasurementDone(std::optional<long long> timeout) {
    logger->trace("Waiting for HVPS Measurement done IRQ");
    auto measurementDoneIrq = static_cast<unsigned>(IUs4OEM::MSINumber::HVPS_MEASUREMENT_DONE);
    this->waitForIrq(measurementDoneIrq, timeout);
}

float Us4OEMImpl::getActualTxFrequency(float frequency) {
    return ius4oem->getOcwsFrequency(frequency);
}
void Us4OEMImpl::setRxSettings(const RxSettings &settings) {
    ius4oem->setRxSettings(settings, false);
}

std::pair<float, float> Us4OEMImpl::getTGCValueRange() const {
    return ius4oem->getTgcValueRange();
}

void Us4OEMImpl::setTxDelaysProfiles(const std::vector<std::pair<size_t, size_t>> &profiles) {
    std::vector<size_t> newProfiles(currentTxDelayProfileIds.size());
    for(const auto &[sequenceId, profileId] : profiles) {
        if(sequenceId > currentTxDelayProfileIds.size()) {
            throw IllegalArgumentException("The sequence with id {} is out of the scope of the "
                                           "currently uploaded scheme (the number of uploaded sequences: {})",
                                                  sequenceId, currentTxDelayProfileIds.size());
        }
        newProfiles.at(sequenceId) = profileId;
    }
    ius4oem->setTxDelays(newProfiles);
    currentTxDelayProfileIds = newProfiles;
}

Us4OEM::Variant Us4OEMImpl::getVariant() {
    const auto &sn = this->serialNumber.get();
    auto variantStr = std::string();
    // us4OEM+
    const std::string pattern2OrdinalStr = "^([a-zA-Z][a-zA-Z\\-]?)([0-9]{10}).*";
    std::regex pattern2OrdinalRegex(pattern2OrdinalStr);
    const std::string pattern4OrdinalStr = "^([a-zA-Z][a-zA-Z\\-]?)([0-9]{12}).*";
    std::regex pattern4OrdinalRegex(pattern4OrdinalStr);
    const size_t OEM_PLUS_SN_2_ORDINAL_SIZE = 12;
    const size_t OEM_PLUS_SN_4_ORDINAL_SIZE = 14;

    std::smatch matches;

    if (sn.empty()) {
        // Legacy us4OEM
        return Us4OEM::Variant::LEGACY;
    }
    else if(sn.size() == OEM_PLUS_SN_2_ORDINAL_SIZE) {
        // us4OEM+
        if(! std::regex_match(sn, matches, pattern2OrdinalRegex) ) {
            throw ::arrus::IllegalStateException("Unrecognized serial number: {}, should have the following pattern: {}.", sn, pattern2OrdinalStr);
        }
        const auto mountingType = matches[1].str();
        const auto number = matches[2].str();

        if (mountingType == "ST" || mountingType == "RA") {
            // Legacy us4OEM+ serial number pattern
            variantStr = number.substr(0, 2);
        } else {
            // Current us4OEM+ serial number pattern
            variantStr = number.substr(8, 2);
        }
    }
    else if (sn.size() == OEM_PLUS_SN_4_ORDINAL_SIZE) {
        if(! std::regex_match(sn, matches, pattern4OrdinalRegex) ) {
            throw ::arrus::IllegalStateException("Unrecognized serial number: {}, should have the following pattern: {}.", sn, pattern4OrdinalStr);
        }
        const auto number = matches[2].str();
        variantStr = number.substr(10, 2);
    }
    else {
        throw ::arrus::IllegalStateException("Unrecognized serial number: {}, should be empty (legacy) or have 12 or 14 characters.", sn);
    }

    const auto variantSymbol = variantStr.at(0);
    if(variantSymbol == '0')  {
        return Us4OEM::Variant::PLUS_RX_32;
    }
    else if(variantSymbol == '1') {
        return Us4OEM::Variant::PLUS_RX_64;
    }
    else if(variantSymbol == '2') {
        return Us4OEM::Variant::PLUS_HF;
    }
    else {
        throw IllegalStateException("Unknown variant for OEM with SN: {}", sn);
    }
}

int64_t Us4OEMImpl::getHVPSTuningInfo() {
    return ius4oem->getHvpsTuningTimestamp();
}


}// namespace arrus::devices
