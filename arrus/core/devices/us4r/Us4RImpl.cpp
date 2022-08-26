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

void Us4RImpl::setVoltage(Voltage voltage) {
    logger->log(LogSeverity::INFO, ::arrus::format("Setting voltage {}", voltage));
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    // Validate.
    auto *device = getDefaultComponent();
    auto voltageRange = device->getAcceptedVoltageRange();

    auto minVoltage = voltageRange.start();
    auto maxVoltage = voltageRange.end();

    if (voltage < minVoltage || voltage > maxVoltage) {
        throw IllegalArgumentException(
            ::arrus::format("Unaccepted voltage '{}', should be in range: [{}, {}]", voltage, minVoltage, maxVoltage));
    }
    hv.value()->setVoltage(voltage);

    //Wait to stabilise voltage output
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    //Verify register
    if (this->getVoltage() != voltage) {
        //throw exception
    }

    float tolerance = 3.0f; // 3V tolerance 

    //Verify measured voltages on HV
    //HVP voltage
    float measured = getMeasuredPVoltage();
    uint8_t retries = 5;

    while ((abs(measured - static_cast<float>(voltage)) > tolerance) && retries--)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        measured = getMeasuredPVoltage();
    }
    if (abs(measured - static_cast<float>(voltage)) > tolerance) {
        disableHV();
        //throw exception
        throw IllegalStateException(
            ::arrus::format("Measured HVP voltage on HV module invalid '{}', should be in range: [{}, {}]",
                measured, (static_cast<float>(voltage) - tolerance), (static_cast<float>(voltage) + tolerance)));
    }
    logger->log(LogSeverity::INFO, ::arrus::format("Measured HVP on HV module = {} V", measured));

    //HVM voltage
    measured = getMeasuredMVoltage();
    retries = 5;

    while ((abs(measured - static_cast<float>(voltage)) > tolerance) && retries--)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        measured = getMeasuredMVoltage();
    }
    if (abs(measured - static_cast<float>(voltage)) > tolerance) {
        disableHV();
        //throw exception
        throw IllegalStateException(
            ::arrus::format("Measured HVM voltage on HV module invalid '{}', should be in range: [{}, {}]",
                measured, (static_cast<float>(voltage) - tolerance), (static_cast<float>(voltage) + tolerance)));
    }
    logger->log(LogSeverity::INFO, ::arrus::format("Measured HVM on HV module= {} V", measured));

    //Verify measured voltages on OEMs
    for (uint8_t i = 0; i < getNumberOfUs4OEMs(); i++) {
        //HVP voltage
        measured = getUCDMeasuredHVPVoltage(i);
        retries = 5;

        while ((abs(measured - static_cast<float>(voltage)) > tolerance) && retries--)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            measured = getUCDMeasuredHVPVoltage(i);
        }
        if (abs(measured - static_cast<float>(voltage)) > tolerance) {
            disableHV();
            //throw exception
            throw IllegalStateException(
                ::arrus::format("Measured HVP voltage on OEM #{} invalid '{}', should be in range: [{}, {}]",
                i, measured, (static_cast<float>(voltage) - tolerance), (static_cast<float>(voltage) + tolerance)));
        }
        logger->log(LogSeverity::INFO, ::arrus::format("Measured HVP on OEM #{} = {} V", i, measured));

        //HVM voltage
        measured = getUCDMeasuredHVMVoltage(i);
        retries = 5;

        while ((abs(measured - static_cast<float>(voltage)) > tolerance) && retries--)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            measured = getUCDMeasuredHVMVoltage(i);
        }
        if (abs(measured - static_cast<float>(voltage)) > tolerance) {
            disableHV();
            //throw exception
            throw IllegalStateException(
                ::arrus::format("Measured HVM voltage on OEM #{} invalid '{}', should be in range: [{}, {}]",
                    i, measured, (static_cast<float>(voltage) - tolerance), (static_cast<float>(voltage) + tolerance)));
        }
        logger->log(LogSeverity::INFO, ::arrus::format("Measured HVM on OEM #{} = {} V", i, measured));
    }
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
        for (auto &us4oem : us4oems) {
            us4oem->getIUs4oem()->DisableInterrupts();
        }
        this->getDefaultComponent()->stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }
    if (this->buffer != nullptr) {
        this->buffer->shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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
    // TODO re-validate the below values.
    float nominalFs = getSamplingFrequency();
    float offset = 300/nominalFs;
    float tgcT = 150/nominalFs; // 150/nominal frequency, the value "150" was determined experimentally.
    return ::arrus::getRange<float>(offset, maxT, tgcT);
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
void Us4RImpl::setLnaGain(uint16 value) {
    ARRUS_ASSERT_RX_SETTINGS_SET();
    auto newRxSettings = RxSettingsBuilder(rxSettings.value()).setLnaGain(value)->build();
    setRxSettings(newRxSettings);
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

}// namespace arrus::devices