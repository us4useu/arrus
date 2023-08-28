#include "Us4RImpl.h"
#include "arrus/core/devices/us4r/validators/RxSettingsValidator.h"
#include "arrus/core/common/interpolate.h"

#include <chrono>
#include <memory>
#include <thread>

#define ARRUS_ASSERT_RX_SETTINGS_SET()                                                                                 \
    if (!rxSettings.has_value()) {                                                                                     \
        throw std::runtime_error("Us4RImpl object has no rx setting set.");                                            \
    }

namespace arrus::devices {

using ::arrus::framework::Buffer;
using ::arrus::framework::DataBufferSpec;
using ::arrus::ops::us4r::Pulse;
using ::arrus::ops::us4r::Rx;
using ::arrus::ops::us4r::Scheme;
using ::arrus::ops::us4r::Tx;
using ::arrus::ops::us4r::TxRxSequence;

UltrasoundDevice *Us4RImpl::getDefaultComponent() {
    // NOTE! The implementation of this function determines
    // validation behaviour of SetVoltage function.
    // The safest option is to prefer using Probe only,
    // with an option to choose us4oem
    // (but the user has to specify it explicitly in settings).
    // Currently there should be no option to set TxRxSequence
    // on an adapter directly.
    if (probe.has_value()) {
        return probe.value().get();
    } else {
        return us4oems[0].get();
    }
}

Us4RImpl::Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::optional<HighVoltageSupplier::Handle> hv,
                   std::vector<unsigned short> channelsMask)
    : Us4R(id), logger{getLoggerFactory()->getLogger()}, us4oems(std::move(us4oems)), hv(std::move(hv)),
      channelsMask(std::move(channelsMask)) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

Us4RImpl::Us4RImpl(const DeviceId &id, Us4RImpl::Us4OEMs us4oems, ProbeAdapterImplBase::Handle &probeAdapter,
                   ProbeImplBase::Handle &probe, std::optional<HighVoltageSupplier::Handle> hv,
                   const RxSettings &rxSettings, std::vector<unsigned short> channelsMask)
    : Us4R(id), logger{getLoggerFactory()->getLogger()}, us4oems(std::move(us4oems)),
      probeAdapter(std::move(probeAdapter)), probe(std::move(probe)), hv(std::move(hv)), rxSettings(rxSettings),
      channelsMask(std::move(channelsMask)) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

std::vector<std::pair <std::string,float>> Us4RImpl::logVoltages(bool isUS4PSC) {
    std::vector<std::pair <std::string,float>> voltages;
    std::pair <std::string,float> temp;
    float voltage;
    // Do not log the voltage measured by US4RPSC, as it may not be correct
    // for this hardware.
    if(!isUS4PSC) {
        //Measure voltages on HV
        voltage = this->getMeasuredPVoltage();
        temp = std::make_pair(std::string("HVP on HV supply"), voltage);
        voltages.push_back(temp);
        voltage = this->getMeasuredMVoltage();
        temp = std::make_pair(std::string("HVM on HV supply"), voltage);
        voltages.push_back(temp);
    }

    //Verify measured voltages on OEMs
    //HV measurements NYI in OEM+
    /*for (uint8_t i = 0; i < getNumberOfUs4OEMs(); i++) {
        voltage = this->getUCDMeasuredHVPVoltage(i);
        temp = std::make_pair(std::string("HVP on OEM#" + std::to_string(i)), voltage);
        voltages.push_back(temp);
        voltage = this->getUCDMeasuredHVMVoltage(i);
        temp = std::make_pair(std::string("HVM on OEM#" + std::to_string(i)), voltage);
        voltages.push_back(temp);
    }*/

    return voltages;
}

void Us4RImpl::checkVoltage(Voltage voltage, float tolerance, int retries, bool isUS4PSC) {
    std::vector<std::pair <std::string,float>> voltages;
    bool fail = true;
    while(retries-- && fail) {
        fail = false;
        voltages = logVoltages(isUS4PSC);
        for(size_t i = 0; i < voltages.size(); i++) {
            if(abs(voltages[i].second - static_cast<float>(voltage)) > tolerance) { 
                fail = true; 
            }
        }
        if(!fail) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    //log last measured voltages
    for(size_t i = 0; i < voltages.size(); i++) {
        logger->log(LogSeverity::INFO, ::arrus::format(voltages[i].first + " = {} V", voltages[i].second));
    }

    if(fail){
        disableHV();
        //find violating voltage
        for(size_t i = 0; i < voltages.size(); i++) {
            if(abs(voltages[i].second - static_cast<float>(voltage)) > tolerance) { 
                throw IllegalStateException(::arrus::format(voltages[i].first + " invalid '{}', should be in range: [{}, {}]",
                voltages[i].second, (static_cast<float>(voltage) - tolerance), (static_cast<float>(voltage) + tolerance)));
            }
        }
    }
}

void Us4RImpl::setVoltage(Voltage voltage) {
    logger->log(LogSeverity::INFO, ::arrus::format("Setting voltage {}", voltage));
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");
    // Validate.
    auto *device = getDefaultComponent();
    auto voltageRange = device->getAcceptedVoltageRange();

    // Note: us4R HV voltage: minimum: 5V, maximum: 90V (this is true for HV256 and US4RPSC).
    auto minVoltage = std::max<unsigned char>(voltageRange.start(), 5);
    auto maxVoltage = std::min<unsigned char>(voltageRange.end(), 90);

    if (voltage < minVoltage || voltage > maxVoltage) {
        throw IllegalArgumentException(
            ::arrus::format("Unaccepted voltage '{}', should be in range: [{}, {}]", voltage, minVoltage, maxVoltage));
    }
    hv.value()->setVoltage(voltage);

    //Wait to stabilise voltage output
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    float tolerance = 4.0f; // 4V tolerance
    int retries = 5;

    //Verify register
    auto &hvModel = this->hv.value()->getModelId();
    bool isUS4PSC = hvModel.getManufacturer() == "us4us" && hvModel.getName() == "us4rpsc";

    if(!isUS4PSC) {
        // Do not check the voltage measured by US4RPSC, as it may not be correct
        // for this hardware.
        Voltage setVoltage = this->getVoltage();
        if (setVoltage != voltage) {
            disableHV();
            throw IllegalStateException(
                ::arrus::format("Voltage set on HV module '{}' does not match requested value: '{}'",setVoltage, voltage));
        }
    }
    else {
        this->logger->log(LogSeverity::INFO,
                          "Skipping voltage verification (measured by HV: "
                          "US4PSC does not provide the possibility to measure the voltage).");
    }

    checkVoltage(voltage, tolerance, retries, isUS4PSC);
}

unsigned char Us4RImpl::getVoltage() {
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    return hv.value()->getVoltage();
}

float Us4RImpl::getMeasuredPVoltage() {
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    return hv.value()->getMeasuredPVoltage();
}

float Us4RImpl::getMeasuredMVoltage() {
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    return hv.value()->getMeasuredMVoltage();
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
    if(this->state == State::STARTED) {
        throw IllegalStateException("You cannot disable HV while the system is running.");
    }
    logger->log(LogSeverity::INFO, "Disabling HV");
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    hv.value()->disable();
}

std::pair<Buffer::SharedHandle, FrameChannelMapping::SharedHandle>
Us4RImpl::upload(const ::arrus::ops::us4r::Scheme &scheme) {
    auto &outputBufferSpec = scheme.getOutputBuffer();
    auto rxBufferNElements = scheme.getRxBufferSize();
    auto &seq = scheme.getTxRxSequence();
    auto workMode = scheme.getWorkMode();

    unsigned hostBufferNElements = outputBufferSpec.getNumberOfElements();

    // Validate input parameters.
    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        IllegalArgumentException("Currently TxRx sequence upload is available for system with probes only."));
    if ((hostBufferNElements % rxBufferNElements) != 0) {
        throw IllegalArgumentException(
            format("The size of the host buffer {} must be equal or a multiple of the size of the rx buffer {}.",
                   hostBufferNElements, rxBufferNElements));
    }
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state == State::STARTED) {
        throw IllegalStateException("The device is running, uploading sequence is forbidden.");
    }
    // Upload and register buffers.
    bool useTriggerSync = workMode == Scheme::WorkMode::HOST || workMode == Scheme::WorkMode::MANUAL;
    auto [rxBuffer, fcm] = uploadSequence(seq, rxBufferNElements, seq.getNRepeats(), useTriggerSync,
                                          scheme.getDigitalDownConversion());
    ARRUS_REQUIRES_TRUE(!rxBuffer->empty(), "Us4R Rx buffer cannot be empty.");

    // Calculate how much of the data each Us4OEM produces.
    auto &element = rxBuffer->getElement(0);
    // a vector, where value[i] contains a size that is produced by a single us4oem.
    std::vector<size_t> us4oemComponentSize(element.getNumberOfUs4oems(), 0);
    int i = 0;
    for (auto &component : element.getUs4oemComponents()) {
        us4oemComponentSize[i++] = component.getSize();
    }
    auto &shape = element.getShape();
    auto dataType = element.getDataType();
    // If the output buffer already exists - remove it.
    if (this->buffer) {
        // The buffer should be already unregistered (after stopping the device).
        this->buffer.reset();
    }
    // Create output buffer.
    this->buffer =
        std::make_shared<Us4ROutputBuffer>(us4oemComponentSize, shape, dataType, hostBufferNElements, stopOnOverflow);
    getProbeImpl()->registerOutputBuffer(this->buffer.get(), rxBuffer, workMode);

    // Note: use only as a marker, that the upload was performed, and there is still some memory to unlock.
    // TODO implement Us4RBuffer move constructor.
    this->us4rBuffer = std::move(rxBuffer);

    return {this->buffer, std::move(fcm)};
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
    for (auto &us4oem : us4oems) {
        us4oem->getIUs4oem()->EnableInterrupts();
    }
    this->getDefaultComponent()->start();
    this->state = State::STARTED;
}

void Us4RImpl::stop() { this->stopDevice(); }

void Us4RImpl::stopDevice() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state != State::STARTED) {
        logger->log(LogSeverity::INFO, "Device Us4R is already stopped.");
    } else {
        logger->log(LogSeverity::DEBUG, "Stopping system.");
        this->getDefaultComponent()->stop();
        for(auto &us4oem: us4oems) {
            us4oem->getIUs4oem()->WaitForPendingTransfers();
            us4oem->getIUs4oem()->WaitForPendingInterrupts();
        }
        // Here all us4R IRQ threads should not work anymore.
        // Cleanup.
        for (auto &us4oem : us4oems) {
            us4oem->getIUs4oem()->DisableInterrupts();
        }
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }
    // TODO: the below should be part of session handler
    if (this->buffer != nullptr) {
        this->buffer->shutdown();
        // We must be sure here, that there is no thread working on the us4rBuffer here.
        if (this->us4rBuffer) {
            getProbeImpl()->unregisterOutputBuffer();
            this->us4rBuffer.reset();
        }
    }
    this->state = State::STOPPED;
}

Us4RImpl::~Us4RImpl() {
    try {
        getDefaultLogger()->log(LogSeverity::DEBUG, "Closing connection with Us4R.");
        this->stopDevice();
        getDefaultLogger()->log(LogSeverity::INFO, "Connection to Us4R closed.");
    } catch(const std::exception &e) {
        std::cerr << "Exception while destroying handle to the Us4R device: " << e.what() << std::endl;
    }
}

std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
Us4RImpl::uploadSequence(const TxRxSequence &seq, uint16 bufferSize, uint16 batchSize, bool triggerSync,
                         const std::optional<ops::us4r::DigitalDownConversion> &ddc) {
    std::vector<TxRxParameters> actualSeq;
    // Convert to intermediate representation (TxRxParameters).
    size_t opIdx = 0;
    for (const auto &txrx : seq.getOps()) {
        auto &tx = txrx.getTx();
        auto &rx = txrx.getRx();

        Interval<uint32> sampleRange(rx.getSampleRange().first, rx.getSampleRange().second);
        Tuple<ChannelIdx> padding({rx.getPadding().first, rx.getPadding().second});

        actualSeq.push_back(TxRxParameters(tx.getAperture(), tx.getDelays(), tx.getExcitation(), rx.getAperture(),
                                           sampleRange, rx.getDownsamplingFactor(), txrx.getPri(), padding));
        ++opIdx;
    }
    return getProbeImpl()->setTxRxSequence(actualSeq, seq.getTgcCurve(), bufferSize, batchSize, seq.getSri(),
                                           triggerSync, ddc);
}

void Us4RImpl::trigger() { this->getDefaultComponent()->syncTrigger(); }

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
    if(y.empty()) {
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
    uint16 offset = 200;
    uint16 tgcT = 150;
    // TODO try avoid converting from samples to time then back to samples?
    uint16 maxNSamples = int16(roundf(maxT*nominalFs));
    // Note: the last TGC sample should be applied before the reception ends.
    // This is to avoid using the same TGC curve between triggers.
    auto values = ::arrus::getRange<uint16>(offset, maxNSamples, tgcT);
    values.push_back(maxNSamples); // TODO(jrozb91) To reconsider (not a full TGC sampling time)
    std::vector<float> time;
    for(auto v: values) {
        time.push_back(v/nominalFs);
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

std::vector<unsigned short> Us4RImpl::getChannelsMask() { return channelsMask; }

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

float Us4RImpl::getCurrentSamplingFrequency() const {return us4oems[0]->getCurrentSamplingFrequency(); }

void Us4RImpl::setHpfCornerFrequency(uint32_t frequency) {
    applyForAllUs4OEMs([frequency](Us4OEM *us4oem) { us4oem->setHpfCornerFrequency(frequency); },
                       "setAfeHpfCornerFrequency");
}

void Us4RImpl::disableHpf() {
    applyForAllUs4OEMs([](Us4OEM *us4oem) { us4oem->disableHpf(); }, "disableHpf");
}

void Us4RImpl::sequencerWriteRegister(uint32_t addr, uint32_t value) {
    us4oems[0]->sequencerWriteRegister(addr, value);
}

uint32_t Us4RImpl::sequencerReadRegister(uint32_t addr) { 
    return us4oems[0]->sequencerReadRegister(addr);
}

uint16_t Us4RImpl::pulserReadRegister(uint8_t sthv, uint16_t addr) {
    return us4oems[0]->pulserReadRegister(sthv, addr);
}

void Us4RImpl::pulserWriteRegister(uint8_t sthv, uint16_t addr, uint16_t reg) {
    us4oems[0]->pulserWriteRegister(sthv, addr, reg);
}

void Us4RImpl::allPulsersWriteRegister(uint16_t addr, uint16_t reg) {
    us4oems[0]->allPulsersWriteRegister(addr, reg);
}

void Us4RImpl::hvpsWriteRegister(uint32_t offset, uint32_t value) {
    for (auto &us4oem : us4oems) {
        us4oem->hvpsWriteRegister(offset, value);
    }
}

uint32_t Us4RImpl::hvpsReadRegister(uint32_t offset) {
    return us4oems[0]->hvpsReadRegister(offset);
}

uint16_t Us4RImpl::getAfe(uint8_t reg) {
    return us4oems[0]->getAfe(reg);
}

void Us4RImpl::setAfe(uint8_t reg, uint16_t val) {
    for (auto &us4oem : us4oems) {
        us4oem->setAfe(reg, val);
    }
}

}// namespace arrus::devices
