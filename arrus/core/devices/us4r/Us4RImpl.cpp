#include "Us4RImpl.h"
#include "arrus/core/devices/us4r/validators/RxSettingsValidator.h"

#include "arrus/core/common/interpolate.h"
#include "arrus/core/devices/probe/ProbeImpl.h"
#include "arrus/core/devices/us4r/mapping/AdaterToUs4OEMMappingConverter.h"
#include "arrus/core/devices/us4r/mapping/ProbeToAdapterMappingConverter.h"
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#define ARRUS_ASSERT_RX_SETTINGS_SET()                                                                                 \
    if (!rxSettings.has_value()) {                                                                                     \
        throw std::runtime_error("Us4RImpl object has no rx setting set.");                                            \
    }

namespace arrus::devices {

using namespace ::arrus::framework;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::session;
using namespace ::arrus::devices::us4r;

Us4RImpl::Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::vector<ProbeSettings> probeSettings,
                   ProbeAdapterSettings probeAdapterSettings, std::vector<HighVoltageSupplier::Handle> hv,
                   const RxSettings &rxSettings, std::vector<std::unordered_set<ChannelIdx>> channelsMask,
                   std::optional<DigitalBackplane::Handle> backplane, std::vector<Bitstream> bitstreams,
                   bool hasIOBitstreamAddressing, const IOSettings &ioSettings)
    : Us4R(id), probeSettings(std::move(probeSettings)), probeAdapterSettings(std::move(probeAdapterSettings)) {
    // Accept empty list of channels masks (no channels masks).
    if(channelsMask.empty()) {
        channelsMask = std::vector{this->probeSettings.size(), std::unordered_set<ChannelIdx>{}};
    }
    this->logger = getLoggerFactory()->getLogger();
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    this->us4oems = std::move(us4oems);
    this->digitalBackplane = std::move(backplane);
    this->hv = std::move(hv);
    this->rxSettings = rxSettings;
    this->channelsMask = std::move(channelsMask);
    this->bitstreams = std::move(bitstreams);
    this->hasIOBitstreamAdressing = hasIOBitstreamAddressing;
    for (size_t i = 0; i < this->probeSettings.size(); ++i) {
        const auto &s = this->probeSettings.at(i).getModel();
        this->probes.push_back(std::make_unique<ProbeImpl>(DeviceId{DeviceType::Probe, Ordinal(i)}, s));
    }

    if (this->hasIOBitstreamAdressing) {
        // Add empty IOBitstream, to use for TX/RX between probe switching.
        getMasterOEM()->getIUs4OEM()->SetWaveformIODriveMode();
        getMasterOEM()->addIOBitstream({0,}, {1,});
    }
    for (auto &bitstream : this->bitstreams) {
        getMasterOEM()->addIOBitstream(bitstream.getLevels(), bitstream.getPeriods());
    }
    this->frameMetadataOEM = getFrameMetadataOEM(ioSettings);
    // Log what probes will be masked
    for (size_t i = 0; i < this->channelsMask.size(); ++i) {
        const auto &mask = this->channelsMask.at(i);
        if (mask.empty()) {
            this->logger->log(LogSeverity::INFO, format("No channel masking applied on Probe:{}", i));
        } else {
            this->logger->log(LogSeverity::INFO,
                              format("The following 'Probe:{}' channels will be masked: {}", i, arrus::toString(mask)));
        }
    }
}

std::vector<std::pair<std::string, float>> Us4RImpl::logVoltages(bool isHV256) {
    std::vector<std::pair<std::string, float>> voltages;
    std::pair<std::string, float> temp;
    float voltage;
    // Do not log the voltage measured by US4RPSC, as it may not be correct
    // for this hardware.
    if (isHV256) {
        //Measure voltages on HV
        voltage = this->getMeasuredPVoltage();
        temp = std::make_pair(std::string("HVP on HV supply"), voltage);
        voltages.push_back(temp);
        voltage = this->getMeasuredMVoltage();
        temp = std::make_pair(std::string("HVM on HV supply"), voltage);
        voltages.push_back(temp);
    }

    auto ver = us4oems[0]->getOemVersion();

    if (ver == 1) {
        //Verify measured voltages on OEMs
        for (uint8_t i = 0; i < getNumberOfUs4OEMs(); i++) {
            voltage = this->getUCDMeasuredHVPVoltage(i);
            temp = std::make_pair(std::string("HVP on OEM#" + std::to_string(i)), voltage);
            voltages.push_back(temp);
            voltage = this->getUCDMeasuredHVMVoltage(i);
            temp = std::make_pair(std::string("HVM on OEM#" + std::to_string(i)), voltage);
            voltages.push_back(temp);
        }
    } else if (ver == 2) {
        //Verify measured voltages on OEM+s
        //Currently OEM+ does not support internal voltage measurement - skip
    }

    return voltages;
}

void Us4RImpl::checkVoltage(Voltage voltage, float tolerance, int retries, bool isHV256) {
    std::vector<std::pair<std::string, float>> voltages;
    bool fail = true;
    while (retries-- && fail) {
        fail = false;
        voltages = logVoltages(isHV256);
        for (size_t i = 0; i < voltages.size(); i++) {
            if (abs(voltages[i].second - static_cast<float>(voltage)) > tolerance) {
                fail = true;
            }
        }
        if (!fail) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    //log last measured voltages
    for (size_t i = 0; i < voltages.size(); i++) {
        logger->log(LogSeverity::INFO, ::arrus::format(voltages[i].first + " = {} V", voltages[i].second));
    }

    if (fail) {
        disableHV();
        //find violating voltage
        for (size_t i = 0; i < voltages.size(); i++) {
            if (abs(voltages[i].second - static_cast<float>(voltage)) > tolerance) {
                throw IllegalStateException(::arrus::format(
                    voltages[i].first + " invalid '{}', should be in range: [{}, {}]", voltages[i].second,
                    (static_cast<float>(voltage) - tolerance), (static_cast<float>(voltage) + tolerance)));
            }
        }
    }
}

void Us4RImpl::setVoltage(Voltage voltage) {
    logger->log(LogSeverity::INFO, ::arrus::format("Setting voltage {}", voltage));
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");
    // Validate.
    Voltage minVoltage = 5, maxVoltage = 90;
    for (const auto &probeSetting : probeSettings) {
        auto probeRange = probeSetting.getModel().getVoltageRange();
        minVoltage = std::max<Voltage>(probeRange.start(), minVoltage);
        maxVoltage = std::min<Voltage>(probeRange.end(), maxVoltage);
    }

    if (voltage < minVoltage || voltage > maxVoltage) {
        throw IllegalArgumentException(
            format("Unaccepted voltage '{}', should be in range: [{}, {}]", voltage, minVoltage, maxVoltage));
    }

    bool isHVPS = true;

    for (uint8_t n = 0; n < hv.size(); n++) {
        auto &hvModel = this->hv[n]->getModelId();
        if (hvModel.getName() != "us4oemhvps") {
            isHVPS = false;
            break;
        }
    }

    if (isHVPS) {
        std::vector<std::future<void>> futures;
        for (uint8_t n = 0; n < hv.size(); n++) {
            futures.push_back(std::async(std::launch::async, &HighVoltageSupplier::setVoltage, hv[n].get(), voltage));
        }
        for (auto &future : futures) {
            future.wait();
        }
    } else {
        for (uint8_t n = 0; n < hv.size(); n++) {
            hv[n]->setVoltage(voltage);
        }
    }

    //Wait to stabilise voltage output
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    float tolerance = 4.0f;// 4V tolerance
    int retries = 5;

    //Verify register
    auto &hvModel = this->hv[0]->getModelId();
    bool isHV256 = hvModel.getManufacturer() == "us4us" && hvModel.getName() == "hv256";

    if (isHV256) {
        // Do not check the voltage measured by US4RPSC, as it may not be correct
        // for this hardware.
        Voltage setVoltage = this->getVoltage();
        if (setVoltage != voltage) {
            disableHV();
            throw IllegalStateException(::arrus::format(
                "Voltage set on HV module '{}' does not match requested value: '{}'", setVoltage, voltage));
        }
    } else {
        this->logger->log(LogSeverity::INFO,
                          "Skipping voltage verification (measured by HV: "
                          "US4PSC does not provide the possibility to measure the voltage).");
    }

    checkVoltage(voltage, tolerance, retries, isHV256);
}

unsigned char Us4RImpl::getVoltage() {
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");
    return hv[0]->getVoltage();
}

float Us4RImpl::getMeasuredPVoltage() {
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");
    return hv[0]->getMeasuredPVoltage();
}

float Us4RImpl::getMeasuredMVoltage() {
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");
    return hv[0]->getMeasuredMVoltage();
}

float Us4RImpl::getUCDMeasuredHVPVoltage(uint8_t oemId) {
    //UCD rail 19 = HVP
    return us4oems[oemId]->getUCDMeasuredVoltage(19);
}

float Us4RImpl::getUCDMeasuredHVMVoltage(uint8_t oemId) {
    //UCD rail 20 = HVM}
    return us4oems[oemId]->getUCDMeasuredVoltage(20);
}

void Us4RImpl::disableHV() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state == State::STARTED) {
        throw IllegalStateException("You cannot disable HV while the system is running.");
    }
    logger->log(LogSeverity::INFO, "Disabling HV");
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");

    for (uint8_t n = 0; n < hv.size(); n++) {
        hv[n]->disable();
    }
}

void Us4RImpl::cleanupBuffers() {
    // The buffer should be already unregistered (after stopping the device).
    this->buffer->shutdown();
    // We must be sure here, that there is no thread working on the us4rBuffer here.
    if (!this->oemBuffers.empty()) {
        unregisterOutputBuffer(false);
        this->oemBuffers.clear();
    }
    this->buffer.reset();
}

std::pair<Buffer::SharedHandle, std::vector<Metadata::SharedHandle>> Us4RImpl::upload(const Scheme &scheme) {
    auto &outputBufferSpec = scheme.getOutputBuffer();
    auto rxBufferSize = scheme.getRxBufferSize();
    auto workMode = scheme.getWorkMode();

    unsigned hostBufferSize = outputBufferSpec.getNumberOfElements();
    // Validate input parameters.
    ARRUS_REQUIRES_TRUE_E(
        (hostBufferSize % rxBufferSize) == 0,
        IllegalArgumentException(
            format("The size of the host buffer {} must be equal or a multiple of the size of the rx buffer {}.",
                   hostBufferSize, rxBufferSize)));

    std::unique_lock<std::mutex> guard(deviceStateMutex);
    ARRUS_REQUIRES_TRUE_E(this->state != State::STARTED,
                          IllegalStateException("The device is running, uploading sequence is forbidden."));
    auto [buffers, fcms] = uploadSequences(scheme.getTxRxSequences(), rxBufferSize, workMode,
                                           scheme.getDigitalDownConversion(), scheme.getConstants());

    // Cleanup.
    // If the output buffer already exists - remove it.
    if (this->buffer) {
        cleanupBuffers();
    }
    // Reset previously set properties for buffer handling.
    for(auto &us4oem: this->us4oems) {
        us4oem->getIUs4OEM()->DisableWaitOnReceiveOverflow();
        us4oem->getIUs4OEM()->DisableWaitOnTransferOverflow();
    }
    // Create output buffer.
    Us4ROutputBufferBuilder builder;
    this->buffer = builder.setStopOnOverflow(stopOnOverflow)
                          .setNumberOfElements(scheme.getOutputBuffer().getNumberOfElements())
                          .setLayoutTo(buffers)
                          .build();
    registerOutputBuffer(this->buffer.get(), buffers, workMode);
    // Note: use only as a marker, that the upload was performed, and there is still some memory to unlock.
    this->oemBuffers = std::move(buffers);
    // Metadata
    std::vector<Metadata::SharedHandle> metadatas;
    for (auto &fcm : fcms) {
        MetadataBuilder metadataBuilder;
        metadataBuilder.add<FrameChannelMapping>("frameChannelMapping", std::move(fcm));
        metadatas.emplace_back(metadataBuilder.buildPtr());
    }
    return {this->buffer, metadatas};
}

void Us4RImpl::start() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    logger->log(LogSeverity::INFO, "Starting us4r.");
    if (this->buffer == nullptr) {
        throw ::arrus::IllegalArgumentException("Call upload function first.");
    }
    if (this->state == State::STARTED) {
        throw ::arrus::IllegalStateException("Device is already running.");
    }
    this->buffer->resetState();
    if (!this->buffer->getOnNewDataCallback()) {
        throw ::arrus::IllegalArgumentException("'On new data callback' is not set.");
    }
    this->state = State::START_IN_PROGRESS;
    for (auto &us4oem : us4oems) {
        us4oem->getIUs4OEM()->EnableInterrupts();
    }
    //  EnableSequencer resets position of the us4oem sequencer.
    for(auto &us4oem: this->us4oems) {
        // Reset tx subsystem pointers.
        us4oem->getIUs4OEM()->EnableTransmit();
        // Reset sequencer pointers.
        us4oem->enableSequencer(true);
    }
    this->getMasterOEM()->start();
    this->state = State::STARTED;
}

void Us4RImpl::stop() { this->stopDevice(); }

void Us4RImpl::stopDevice() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state != State::STARTED) {
        logger->log(LogSeverity::INFO, "Device Us4R is already stopped.");
    } else {
        this->state = State::STOP_IN_PROGRESS;
        logger->log(LogSeverity::DEBUG, "Stopping system.");
        this->getMasterOEM()->stop();
        try {
            for (auto &us4oem : us4oems) {
                us4oem->getIUs4OEM()->WaitForPendingTransfers();
                us4oem->getIUs4OEM()->WaitForPendingInterrupts();
            }
        }
        catch(const std::exception &e) {
            logger->log(
                LogSeverity::WARNING,
                arrus::format("Error on waiting for pending interrupts and transfers: {}", e.what()));
        }
        // Here all us4R IRQ threads should not work anymore.
        // Cleanup.
        for (auto &us4oem : us4oems) {
            us4oem->getIUs4OEM()->DisableInterrupts();
        }
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }
    this->state = State::STOPPED;
}

Us4RImpl::~Us4RImpl() {
    try {
        getDefaultLogger()->log(LogSeverity::DEBUG, "Closing connection with Us4R.");
        this->stopDevice();
        // TODO: the below should be part of session handler
        if (this->buffer) {
            cleanupBuffers();
        }
        getDefaultLogger()->log(LogSeverity::INFO, "Connection to Us4R closed.");
    } catch (const std::exception &e) {
        std::cerr << "Exception while destroying handle to the Us4R device: " << e.what() << std::endl;
    }
}

std::pair<std::vector<Us4OEMBuffer>, std::vector<FrameChannelMapping::Handle>>
Us4RImpl::uploadSequences(const std::vector<TxRxSequence> &sequences, uint16 bufferSize, Scheme::WorkMode workMode,
                          const std::optional<DigitalDownConversion> &ddc,
                          const std::vector<NdArray> &txDelayProfiles) {
    // Convert to intermediate representation (TxRxParameters).
    auto noems = ARRUS_SAFE_CAST(this->us4oems.size(), Ordinal);
    auto nSequences = ARRUS_SAFE_CAST(sequences.size(), SequenceId);
    // Sequence id -> converter
    std::vector<ProbeToAdapterMappingConverter> probe2Adapter;
    // Sequence id -> converter
    std::vector<AdapterToUs4OEMMappingConverter> adapter2OEM;
    // Convert API sequences to internal representation.
    std::vector<TxRxParametersSequence> seqs = convertToInternalSequences(sequences);
    // Initialize converters.
    auto oemMappings = getOEMMappings();
    for (SequenceId sId = 0; sId < nSequences; ++sId) {
        auto s = seqs.at(sId);
        const auto &txProbeId = s.getTxProbeId();
        const auto &rxProbeId = s.getRxProbeId();
        const auto &txProbeSettings = probeSettings.at(txProbeId.getOrdinal());
        const auto &rxProbeSettings = probeSettings.at(rxProbeId.getOrdinal());
        const auto &txProbeMask = channelsMask.at(txProbeId.getOrdinal());
        const auto &rxProbeMask = channelsMask.at(rxProbeId.getOrdinal());
        auto nAdapterChannels = probeAdapterSettings.getNumberOfChannels();
        // Find the correct probe TX, RX ordinal
        probe2Adapter.emplace_back(
            txProbeId, rxProbeId, txProbeSettings, rxProbeSettings, txProbeMask, rxProbeMask, nAdapterChannels);
        adapter2OEM.emplace_back(
            probeAdapterSettings, noems, oemMappings, frameMetadataOEM,
            // NOTE assuming that all OEMs have the same number of RX channels
            getMasterOEM()->getDescriptor().getNRxChannels());
    }

    using OEMSequences = AdapterToUs4OEMMappingConverter::OEMSequences;
    // OEM -> list of sequences to upload on the OEM
    auto sequencesByOEM = std::vector{noems, OEMSequences{}};
    // Convert probe sequence -> OEM Sequences
    for (SequenceId sId = 0; sId < nSequences; ++sId) {
        const auto &s = seqs.at(sId);
        auto [as, adapterDelays] = probe2Adapter.at(sId).convert(sId, s, txDelayProfiles);
        auto [oemSeqs, oemDelays] = adapter2OEM.at(sId).convert(sId, as, adapterDelays);
        for (Ordinal oem = 0; oem < noems; ++oem) {
            sequencesByOEM.at(oem).emplace_back(std::move(oemSeqs.at(oem)));
        }
    }
    std::vector<Us4OEMBuffer> buffers;
    // Sequence id -> the probe-level FCM.
    std::vector<FrameChannelMapping::Handle> fcms;
    // Sequence id, OEM -> FCM
    std::vector<std::vector<FrameChannelMapping::Handle>> oemsFCMs;
    for (SequenceId sId = 0; sId < nSequences; ++sId) { oemsFCMs.emplace_back(); }
    for (Ordinal oem = 0; oem < noems; ++oem) {
        // TODO Consider implementing dynamic change of delay profiles
        auto uploadResult =
            us4oems.at(oem)->upload(sequencesByOEM.at(oem), bufferSize, workMode, ddc, std::vector<NdArray>{});
        buffers.emplace_back(uploadResult.getBufferDescription());
        auto oemFCM = uploadResult.acquireFCMs();
        for (SequenceId sId = 0; sId < nSequences; ++sId) {
            oemsFCMs.at(sId).emplace_back(std::move(oemFCM.at(sId)));
        }
    }
    // Convert FCMs to probe-level apertures.
    for (SequenceId sId = 0; sId < nSequences; ++sId) {
        auto adapterFCM = adapter2OEM.at(sId).convert(oemsFCMs.at(sId));
        auto probeFCM = probe2Adapter.at(sId).convert(adapterFCM);
        fcms.emplace_back(std::move(probeFCM));
    }
    return std::make_pair(std::move(buffers), std::move(fcms));
}

TxRxParameters Us4RImpl::createBitstreamSequenceSelectPreamble(const TxRxSequence &sequence) {
    ARRUS_REQUIRES_TRUE(!sequence.getOps().empty(), "The sequence should have at least one TX/RX defined.");
    // Make sure that all ops in the sequence use the same bitstream
    auto rxProbe = sequence.getRxProbeId().getOrdinal();
    auto txProbe = sequence.getTxProbeId().getOrdinal();
    ARRUS_REQUIRES_TRUE_IAE(
        probeSettings.at(rxProbe).getBitstreamId() == probeSettings.at(txProbe).getBitstreamId(),
        format("All probes used within a single sequence should be connected with the same bitstream ID, "
               "incorrect probes: {}, {}",
               txProbe, rxProbe));
    auto bitstreamId = probeSettings.at(rxProbe).getBitstreamId();
    TxRxParametersBuilder preambleBuilder(sequence.getOps().at(0));
    // emplace a single, short op, for bitstream pattern switching only
    // No TX/RX
    preambleBuilder.convertToNOP();
    // Bitstream only.
    preambleBuilder.setBitstreamId(bitstreamId);
    // NOTE: 2ms is an aribtrary value (should work at least for 576-channel MUX board).
    preambleBuilder.setPri(2000e-6f);
    return preambleBuilder.build();
}

std::vector<TxRxParametersSequence> Us4RImpl::convertToInternalSequences(const std::vector<TxRxSequence> &sequences) {
    std::vector<TxRxParametersSequence> result;
    std::optional<BitstreamId> currentBitstreamId = std::nullopt;
    for(const auto &sequence: sequences) {
        TxRxParametersSequenceBuilder sequenceBuilder;
        sequenceBuilder.setCommon(sequence);
        if (hasIOBitstreamAdressing) {
            auto preamble = createBitstreamSequenceSelectPreamble(sequence);
            if(!currentBitstreamId.has_value() || currentBitstreamId.value() != preamble.getBitstreamId()) {
                // Use the preamble only when the Bitstream Id changes.
                // For example, if the same bitstream id is used in two consecutive sequences -- skip the preamble.
                sequenceBuilder.addEntry(preamble);
                currentBitstreamId = preamble.getBitstreamId();
            }
        }
        auto rxDelay = getRxDelay(sequence);
        for (const auto &txrx : sequence.getOps()) {
            TxRxParametersBuilder builder(txrx);
            builder.setRxDelay(rxDelay);
            if (hasIOBitstreamAdressing) {
                builder.setBitstreamId(BitstreamId(0));
            }
            sequenceBuilder.addEntry(builder.build());
        }
        result.emplace_back(sequenceBuilder.build());
    }
    return result;
}

void Us4RImpl::trigger() { this->getMasterOEM()->syncTrigger(); }

// AFE parameter setters.
void Us4RImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints) { setTgcCurve(tgcCurvePoints, true); }

void Us4RImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value())
                             .setTgcSamples(tgcCurvePoints)
                             ->setApplyTgcCharacteristic(applyCharacteristic)
                             ->build();
    setRxSettings(newRxSettings);
}

void Us4RImpl::setTgcCurve(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic) {
    ARRUS_REQUIRES_TRUE(t.size() == y.size(), "TGC sample values t and y should have the same size.");
    if (y.empty()) {
        // Turn off TGC
        setTgcCurve(y, applyCharacteristic);
    } else {
        auto timeStartIt = std::min_element(std::begin(t), std::end(t));
        auto timeEndIt = std::max_element(std::begin(t), std::end(t));

        auto timeEnd = *timeEndIt;

        auto valueStart = y[std::distance(std::begin(t), timeStartIt)];
        auto valueEnd = y[std::distance(std::begin(t), timeEndIt)];

        std::vector<float> hardwareTgcSamplingPoints = getTgcCurvePoints(timeEnd);
        auto tgcValues = ::arrus::interpolate1d<float>(t, y, hardwareTgcSamplingPoints, valueStart, valueEnd);
        setTgcCurve(tgcValues, applyCharacteristic);
    }
}

std::vector<float> Us4RImpl::getTgcCurvePoints(float maxT) const {
    // TODO(jrozb91) To reconsider below.
    float nominalFs = getSamplingFrequency();
    uint16 offset = 359;
    uint16 tgcT = 153;
    // TODO try avoid converting from samples to time then back to samples?
    uint16 maxNSamples = int16(roundf(maxT * nominalFs));
    // Note: the last TGC sample should be applied before the reception ends.
    // This is to avoid using the same TGC curve between triggers.
    auto values = ::arrus::getRange<uint16>(offset, maxNSamples, tgcT);
    values.push_back(maxNSamples);// TODO(jrozb91) To reconsider (not a full TGC sampling time)
    std::vector<float> time;
    for (auto v : values) {
        time.push_back(v / nominalFs);
    }
    return time;
}

void Us4RImpl::setRxSettings(const RxSettings &settings) {
    RxSettingsValidator validator;
    validator.validate(settings);
    validator.throwOnErrors();

    std::unique_lock<std::mutex> guard(afeParamsMutex);
    bool isStateInconsistent = false;
    try {
        for (auto &us4oem : us4oems) {
            us4oem->setRxSettings(settings);
            // At least one us4OEM has been updated.
            isStateInconsistent = true;
        }
        isStateInconsistent = false;
        this->rxSettings = settings;
    } catch (...) {
        if (isStateInconsistent) {
            logger->log(LogSeverity::ERROR,
                        "Us4R AFE parameters are in inconsistent state: some of the us4OEM modules "
                        "were not properly configured.");
        }
        throw;
    }
}

void Us4RImpl::setPgaGain(uint16 value) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value()).setPgaGain(value)->build();
    setRxSettings(newRxSettings);
}
uint16 Us4RImpl::getPgaGain() {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    return this->rxSettings.value().getPgaGain();
}
void Us4RImpl::setLnaGain(uint16 value) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value()).setLnaGain(value)->build();
    setRxSettings(newRxSettings);
}
uint16 Us4RImpl::getLnaGain() {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    return this->rxSettings.value().getLnaGain();
}
void Us4RImpl::setLpfCutoff(uint32 value) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value()).setLpfCutoff(value)->build();
    setRxSettings(newRxSettings);
}
void Us4RImpl::setDtgcAttenuation(std::optional<uint16> value) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value()).setDtgcAttenuation(value)->build();
    setRxSettings(newRxSettings);
}
void Us4RImpl::setActiveTermination(std::optional<uint16> value) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value()).setActiveTermination(value)->build();
    setRxSettings(newRxSettings);
}

uint8_t Us4RImpl::getNumberOfUs4OEMs() { return static_cast<uint8_t>(us4oems.size()); }

void Us4RImpl::setTestPattern(Us4OEM::RxTestPattern pattern) {
    // TODO make the below exception safe
    for (auto &us4oem : us4oems) {
        us4oem->setTestPattern(pattern);
    }
}

float Us4RImpl::getSamplingFrequency() const { return (float) us4oems[0]->getSamplingFrequency(); }

void Us4RImpl::checkState() const {
    for (auto &us4oem : us4oems) {
        us4oem->checkState();
    }
}

void Us4RImpl::setStopOnOverflow(bool value) {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state != State::STOPPED) {
        logger->log(LogSeverity::INFO,
                    "The StopOnOverflow property can be set "
                    "only when the device is stopped.");
    }
    this->stopOnOverflow = value;
}

bool Us4RImpl::isStopOnOverflow() const { return stopOnOverflow; }

void Us4RImpl::applyForAllUs4OEMs(const std::function<void(Us4OEM *us4oem)> &func, const std::string &funcName) {
    bool isConsistent = false;
    try {
        for (auto &us4oem : us4oems) {
            func(us4oem.get());
            // At least one us4OEM has been updated, some us4OEMs have been already updated.
            isConsistent = true;
        }
        isConsistent = false;
    } catch (...) {
        if (isConsistent) {
            logger->log(LogSeverity::ERROR,
                        format("Error while calling '{}': the function was not applied "
                               "correctly for all us4OEMs.",
                               funcName));
        }
        throw;
    }
}

void Us4RImpl::setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                           size_t nCoefficients) {
    applyForAllUs4OEMs(
        [demodulationFrequency, decimationFactor, firCoefficients, nCoefficients](Us4OEM *us4oem) {
            us4oem->setAfeDemod(demodulationFrequency, decimationFactor, firCoefficients, nCoefficients);
        },
        "setAfeDemod");
}

void Us4RImpl::disableAfeDemod() {
    applyForAllUs4OEMs([](Us4OEM *us4oem) { us4oem->disableAfeDemod(); }, "disableAfeDemod");
}

float Us4RImpl::getCurrentSamplingFrequency() const { return us4oems[0]->getCurrentSamplingFrequency(); }

void Us4RImpl::setHpfCornerFrequency(uint32_t frequency) {
    applyForAllUs4OEMs([frequency](Us4OEM *us4oem) { us4oem->setHpfCornerFrequency(frequency); },
                       "setAfeHpfCornerFrequency");
}

void Us4RImpl::disableHpf() {
    applyForAllUs4OEMs([](Us4OEM *us4oem) { us4oem->disableHpf(); }, "disableHpf");
}

uint16_t Us4RImpl::getAfe(uint8_t reg) { return us4oems[0]->getAfe(reg); }

void Us4RImpl::setAfe(uint8_t reg, uint16_t val) {
    for (auto &us4oem : us4oems) {
        us4oem->setAfe(reg, val);
    }
}

void Us4RImpl::registerOutputBuffer(Us4ROutputBuffer *outputBuffer, const std::vector<Us4OEMBuffer> &srcBuffers,
                                    Scheme::WorkMode workMode) {
    Ordinal o = 0;

    if (transferRegistrar.size() < us4oems.size()) {
        transferRegistrar.resize(us4oems.size());
    }
    for (auto &us4oem : us4oems) {
        auto us4oemBuffer = srcBuffers.at(o);
        this->registerOutputBuffer(outputBuffer, us4oemBuffer, us4oem.get(), workMode);
        ++o;
    }
}

/**
 * - This function assumes, that the size of output buffer (number of elements)
 *  is a multiple of number of us4oem elements.
 * - this function will not schedule data transfer when the us4oem element size is 0.
 */
void Us4RImpl::registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc, Us4OEMImplBase *us4oem,
                                    Scheme::WorkMode workMode) {
    auto us4oemOrdinal = us4oem->getDeviceId().getOrdinal();
    auto ius4oem = us4oem->getIUs4OEM();
    const auto nElementsSrc = bufferSrc.getNumberOfElements();
    const size_t nElementsDst = bufferDst->getNumberOfElements();
    size_t elementSize = getUniqueUs4OEMBufferElementSize(bufferSrc);
    if (elementSize == 0) {
        return;
    }
    if (transferRegistrar[us4oemOrdinal]) {
        transferRegistrar[us4oemOrdinal].reset();
    }
    transferRegistrar[us4oemOrdinal] = std::make_shared<Us4OEMDataTransferRegistrar>(
        bufferDst, bufferSrc, us4oem, us4oem->getDescriptor().getMaxTransferSize());
    transferRegistrar[us4oemOrdinal]->registerTransfers();
    // Register buffer element release functions.
    bool isMaster = us4oem->getDeviceId().getOrdinal() == this->getMasterOEM()->getDeviceId().getOrdinal();
    size_t nRepeats = nElementsDst / nElementsSrc;
    uint16 startFiring = 0;
    for (size_t i = 0; i < bufferSrc.getNumberOfElements(); ++i) {
        auto &srcElement = bufferSrc.getElement(i);
        uint16 endFiring = srcElement.getFiring();
        for (size_t j = 0; j < nRepeats; ++j) {
            std::function<void()> releaseFunc = createReleaseCallback(workMode, startFiring, endFiring);
            bufferDst->registerReleaseFunction(j * nElementsSrc + i, releaseFunc);
        }
        startFiring = endFiring + 1;
    }
    // Overflow handling
    ius4oem->RegisterReceiveOverflowCallback(createOnReceiveOverflowCallback(workMode, bufferDst, isMaster));
    ius4oem->RegisterTransferOverflowCallback(createOnTransferOverflowCallback(workMode, bufferDst, isMaster));
    // Work mode specific initialization
    if (workMode == ops::us4r::Scheme::WorkMode::SYNC) {
        ius4oem->EnableWaitOnReceiveOverflow();
        ius4oem->EnableWaitOnTransferOverflow();
    }
}

size_t Us4RImpl::getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const {
    std::unordered_set<size_t> sizes;
    for (auto &element : us4oemBuffer.getElements()) {
        sizes.insert(element.getSize());
    }
    if (sizes.size() > 1) {
        throw ArrusException("Each us4oem buffer element should have the same size.");
    }
    // This is the size of a single element produced by this us4oem.
    const size_t elementSize = *std::begin(sizes);
    return elementSize;
}

void Us4RImpl::unregisterOutputBuffer(bool cleanupSequencer) {
    if(transferRegistrar.empty()) {
        return;
    }
    for (Ordinal i = 0; i < us4oems.size(); ++i) {
        if(transferRegistrar[i]) {
            transferRegistrar[i]->unregisterTransfers(cleanupSequencer);
        }
    }
}

std::function<void()> Us4RImpl::createReleaseCallback(Scheme::WorkMode workMode, uint16 startFiring, uint16 endFiring) {

    switch (workMode) {
    case Scheme::WorkMode::HOST:// Automatically generate new trigger after releasing all elements.
        return [this, startFiring, endFiring]() {
            for (int i = (int) us4oems.size() - 1; i >= 0; --i) {
                us4oems[i]->getIUs4OEM()->MarkEntriesAsReadyForReceive(startFiring, endFiring);
                us4oems[i]->getIUs4OEM()->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
            }
            if (this->state != State::STOP_IN_PROGRESS && this->state != State::STOPPED) {
                getMasterOEM()->syncTrigger();
            }
        };
    case Scheme::WorkMode::ASYNC: // Trigger generator: us4R
    case Scheme::WorkMode::SYNC:  // Trigger generator: us4R
    case Scheme::WorkMode::MANUAL:// Trigger generator: external (e.g. user)
        return [this, startFiring, endFiring]() {
            for (int i = (int) us4oems.size() - 1; i >= 0; --i) {
                us4oems[i]->getIUs4OEM()->MarkEntriesAsReadyForReceive(startFiring, endFiring);
                us4oems[i]->getIUs4OEM()->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
            }
        };
    default: throw ::arrus::IllegalArgumentException("Unsupported work mode.");
    }
}

std::function<void()> Us4RImpl::createOnReceiveOverflowCallback(Scheme::WorkMode workMode,
                                                                Us4ROutputBuffer *outputBuffer, bool isMaster) {

    using namespace std::chrono_literals;
    switch (workMode) {
    case Scheme::WorkMode::SYNC:
        return [this, outputBuffer, isMaster]() {
            try {
                this->logger->log(LogSeverity::WARNING, "Detected RX data overflow.");
                size_t nElements = outputBuffer->getNumberOfElements();
                // Wait for all elements to be released by the user.
                while (nElements != outputBuffer->getNumberOfElementsInState(framework::BufferElement::State::FREE)) {
                    std::this_thread::sleep_for(1ms);
                    if (this->state != State::STARTED) {
                        // Device is no longer running, exit gracefully.
                        return;
                    }
                }
                // Inform about free elements only once, in the master's callback.
                if (isMaster) {
                    for (int i = (int) us4oems.size() - 1; i >= 0; --i) {
                        us4oems[i]->getIUs4OEM()->SyncReceive();
                    }
                }
                outputBuffer->runOnOverflowCallback();
            } catch (const std::exception &e) {
                logger->log(LogSeverity::ERROR, format("RX overflow callback exception: ", e.what()));
            } catch (...) { logger->log(LogSeverity::ERROR, "RX overflow callback exception: unknown"); }
        };
    case Scheme::WorkMode::ASYNC:
    case Scheme::WorkMode::HOST:
    case Scheme::WorkMode::MANUAL:
        return [this, outputBuffer]() {
            try {
                if (outputBuffer->isStopOnOverflow()) {
                    this->logger->log(LogSeverity::ERROR, "Rx data overflow, stopping the device.");
                    this->getMasterOEM()->stop();
                    outputBuffer->markAsInvalid();
                } else {
                    this->logger->log(LogSeverity::WARNING, "Rx data overflow ...");
                    outputBuffer->runOnOverflowCallback();
                }
            } catch (const std::exception &e) {
                logger->log(LogSeverity::ERROR, format("RX overflow callback exception: ", e.what()));
            } catch (...) { logger->log(LogSeverity::ERROR, "RX overflow callback exception: unknown"); }
        };
    default: throw ::arrus::IllegalArgumentException("Unsupported work mode.");
    }
}

std::function<void()> Us4RImpl::createOnTransferOverflowCallback(Scheme::WorkMode workMode,
                                                                 Us4ROutputBuffer *outputBuffer, bool isMaster) {

    using namespace std::chrono_literals;
    switch (workMode) {
    case Scheme::WorkMode::SYNC:
        return [this, outputBuffer, isMaster]() {
            try {
                this->logger->log(LogSeverity::WARNING, "Detected host data overflow.");
                size_t nElements = outputBuffer->getNumberOfElements();
                // Wait for all elements to be released by the user.
                while (nElements != outputBuffer->getNumberOfElementsInState(framework::BufferElement::State::FREE)) {
                    std::this_thread::sleep_for(1ms);
                    if (this->state != State::STARTED) {
                        // Device is no longer running, exit gracefully.
                        return;
                    }
                }
                // Inform about free elements only once, in the master's callback.
                if (isMaster) {
                    for (int i = (int) us4oems.size() - 1; i >= 0; --i) {
                        us4oems[i]->getIUs4OEM()->SyncTransfer();
                    }
                }
                outputBuffer->runOnOverflowCallback();
            } catch (const std::exception &e) {
                logger->log(LogSeverity::ERROR, format("Host overflow callback exception: ", e.what()));
            } catch (...) { logger->log(LogSeverity::ERROR, "Host overflow callback exception: unknown"); }
        };
    case Scheme::WorkMode::ASYNC:
    case Scheme::WorkMode::HOST:
    case Scheme::WorkMode::MANUAL:
        return [this, outputBuffer]() {
            try {
                if (outputBuffer->isStopOnOverflow()) {
                    this->logger->log(LogSeverity::ERROR, "Host data overflow, stopping the device.");
                    this->getMasterOEM()->stop();
                    outputBuffer->markAsInvalid();
                } else {
                    outputBuffer->runOnOverflowCallback();
                    this->logger->log(LogSeverity::WARNING, "Host data overflow ...");
                }
            } catch (const std::exception &e) {
                logger->log(LogSeverity::ERROR, format("Host overflow callback exception: ", e.what()));
            } catch (...) { logger->log(LogSeverity::ERROR, "Host overflow callback exception: unknown"); }
        };
    default: throw ::arrus::IllegalArgumentException("Unsupported work mode.");
    }
}

const char *Us4RImpl::getBackplaneSerialNumber() {
    if (!this->digitalBackplane.has_value()) {
        throw arrus::IllegalArgumentException("No backplane defined.");
    }
    return this->digitalBackplane->get()->getSerialNumber();
}

const char *Us4RImpl::getBackplaneRevision() {
    if (!this->digitalBackplane.has_value()) {
        throw arrus::IllegalArgumentException("No backplane defined.");
    }
    return this->digitalBackplane->get()->getRevisionNumber();
}

void Us4RImpl::setParameters(const Parameters &params) {
    for (auto &item : params.items()) {
        auto &key = item.first;
        auto value = item.second;
        logger->log(LogSeverity::INFO, format("Setting value {} to {}", value, key));
        if (key != "/sequence:0/txFocus") {
            throw ::arrus::IllegalArgumentException("Currently Us4R supports only sequence:0/txFocus parameter.");
        }
        this->us4oems[0]->getIUs4OEM()->TriggerStop();
        try {
            for (auto &us4oem : us4oems) {
                us4oem->getIUs4OEM()->SetTxDelays(value);
            }
        } catch (...) {
            // Try resume.
            this->us4oems[0]->getIUs4OEM()->TriggerStart();
            throw;
        }
        // Everything OK, resume.
        this->us4oems[0]->getIUs4OEM()->TriggerStart();
    }
}

BitstreamId Us4RImpl::addIOBitstream(const std::vector<uint8_t> &levels, const std::vector<uint16_t> &periods) {
    return this->us4oems[0]->addIOBitstream(levels, periods);
}

void Us4RImpl::setIOBitstream(BitstreamId bitstreamId, const std::vector<uint8_t> &levels,
                              const std::vector<uint16_t> &periods) {
    this->us4oems[0]->setIOBitstream(bitstreamId, levels, periods);
}

std::vector<std::vector<uint8_t>> Us4RImpl::getOEMMappings() const {
    std::vector<std::vector<uint8_t>> mappings;
    for (auto &us4oem : us4oems) {
        mappings.push_back(us4oem->getChannelMapping());
    }
    return mappings;
}

std::optional<Ordinal> Us4RImpl::getFrameMetadataOEM(const IOSettings &settings) {
    if (!settings.hasFrameMetadataCapability()) {
        return std::nullopt;
    } else {
        std::unordered_set<Ordinal> oems = settings.getFrameMetadataCapabilityOEMs();
        if (oems.size() != 1) {
            throw ::arrus::IllegalArgumentException("Exactly one OEM should be set for the pulse counter capability.");
        } else {
            // Only a single OEM.
            return *std::begin(oems);
        }
    }
}

std::vector<unsigned short> Us4RImpl::getChannelsMask(Ordinal probeNumber) {
    const auto &mask = channelsMask.at(probeNumber);
    std::vector<unsigned short> vec(std::begin(mask), std::end(mask));
    std::sort(std::begin(vec), std::end(vec));
    return vec;
}

int Us4RImpl::getNumberOfProbes() const { return probeSettings.size(); }

/**
 * Calculates RX delay as the maximum TX delay of the sequence + burst time
 * Only TX delays from the active (aperture) elements are considered.
 * If the given sequence does not perform TX, this method returns 0.
 *
 * @return rx delay [s]
 */
float Us4RImpl::getRxDelay(const TxRxSequence &sequence) {
    std::vector<float> opDelays;
    for(const auto &op: sequence.getOps()) {
        std::vector<float> delays;
        for(size_t i = 0; i < op.getTx().getAperture().size(); ++i) {
            if(op.getTx().getAperture()[i]) {
                delays.push_back(op.getTx().getDelays()[i]);
            }
        }
        if(!delays.empty()) {
            float txDelay = *std::max_element(std::begin(delays), std::end(delays));
            // burst time
            float frequency = op.getTx().getExcitation().getCenterFrequency();
            float nPeriods = op.getTx().getExcitation().getNPeriods();
            float burstTime = 1.0f/frequency*nPeriods;
            // Total rx delay
            opDelays.push_back(txDelay + burstTime);
        }
    }
    if(!opDelays.empty()) {
        return *std::max_element(std::begin(opDelays), std::end(opDelays));
    }
    else {
        // No TX
        return 0.0f;
    }
}

std::pair<std::shared_ptr<Buffer>, std::shared_ptr<session::Metadata>>
Us4RImpl::setSubsequence(uint16_t, uint16_t, const std::optional<float> &) {
    throw std::runtime_error("setSubsequence not yet implemented.");
}


}// namespace arrus::devices
