#include "Us4RImpl.h"
#include "arrus/core/devices/us4r/validators/RxSettingsValidator.h"

#include "arrus/core/common/interpolate.h"

#include <chrono>
#include <memory>
#include <thread>
#include <future>

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

Us4RImpl::Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::vector<HighVoltageSupplier::Handle> hv,
                   std::vector<unsigned short> channelsMask, std::optional<DigitalBackplane::Handle> backplane)
    : Us4R(id), logger{getLoggerFactory()->getLogger()}, us4oems(std::move(us4oems)),
      digitalBackplane(std::move(backplane)),
      hv(std::move(hv)),
      channelsMask(std::move(channelsMask))
{
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

Us4RImpl::Us4RImpl(const DeviceId &id, Us4RImpl::Us4OEMs us4oems, ProbeAdapterImplBase::Handle &probeAdapter,
                   ProbeImplBase::Handle &probe, std::vector<HighVoltageSupplier::Handle> hv,
                   const RxSettings &rxSettings, std::vector<unsigned short> channelsMask,
                   std::optional<DigitalBackplane::Handle> backplane)
    : Us4R(id), logger{getLoggerFactory()->getLogger()}, us4oems(std::move(us4oems)),
      probeAdapter(std::move(probeAdapter)), probe(std::move(probe)),
      digitalBackplane(std::move(backplane)),
      hv(std::move(hv)),
      rxSettings(rxSettings),
      channelsMask(std::move(channelsMask))
{
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

std::vector<std::pair <std::string,float>> Us4RImpl::logVoltages(bool isHV256) {
    std::vector<std::pair <std::string,float>> voltages;
    std::pair <std::string,float> temp;
    float voltage;
    // Do not log the voltage measured by US4RPSC, as it may not be correct
    // for this hardware.
    if(isHV256) {
        //Measure voltages on HV
        voltage = this->getMeasuredPVoltage();
        temp = std::make_pair(std::string("HVP on HV supply"), voltage);
        voltages.push_back(temp);
        voltage = this->getMeasuredMVoltage();
        temp = std::make_pair(std::string("HVM on HV supply"), voltage);
        voltages.push_back(temp);
    }

    //Verify measured voltages on OEMs
    auto isUs4OEMPlus = this->isUs4OEMPlus();
    if(!isUs4OEMPlus || (isUs4OEMPlus && !isHV256)) {
        for (uint8_t i = 0; i < getNumberOfUs4OEMs(); i++) {
            voltage = this->getMeasuredHVPVoltage(i);
            temp = std::make_pair(std::string("HVP on OEM#" + std::to_string(i)), voltage);
            voltages.push_back(temp);
            voltage = this->getMeasuredHVMVoltage(i);
            temp = std::make_pair(std::string("HVM on OEM#" + std::to_string(i)), voltage);
            voltages.push_back(temp);
        }
    }
    return voltages;
}

void Us4RImpl::checkVoltage(Voltage voltage, float tolerance, int retries, bool isHV256) {
    std::vector<std::pair <std::string,float>> voltages;
    bool fail = true;
    while(retries-- && fail) {
        fail = false;
        voltages = logVoltages(isHV256);
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

    bool isHVPS = true;

    for(uint8_t n = 0; n < hv.size(); n++) {
        auto &hvModel = this->hv[n]->getModelId();
        if(hvModel.getName() != "us4oemhvps") {
            isHVPS = false;
            break;
        }
    }

    if(isHVPS) {
        std::vector<std::future<void>> futures;
        for (uint8_t n = 0; n < hv.size(); n++) {
            futures.push_back(std::async(std::launch::async, &HighVoltageSupplier::setVoltage, hv[n].get(), voltage));
        }
        for (auto &future : futures) {
            future.wait();
        }
    }
    else {
        for(uint8_t n = 0; n < hv.size(); n++) {
            hv[n]->setVoltage(voltage);
        }
    }


    //Wait to stabilise voltage output
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    float tolerance = 4.0f; // 4V tolerance
    int retries = 5;

    //Verify register
    auto &hvModel = this->hv[0]->getModelId();
    bool isHV256 = hvModel.getManufacturer() == "us4us" && hvModel.getName() == "hv256";

    if(isHV256) {
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

float Us4RImpl::getMeasuredHVPVoltage(uint8_t oemId) {
    return us4oems[oemId]->getMeasuredHVPVoltage();
}

float Us4RImpl::getMeasuredHVMVoltage(uint8_t oemId) {
    return us4oems[oemId]->getMeasuredHVMVoltage();
}

void Us4RImpl::disableHV() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if(this->state == State::STARTED) {
        throw IllegalStateException("You cannot disable HV while the system is running.");
    }
    logger->log(LogSeverity::INFO, "Disabling HV");
    ARRUS_REQUIRES_TRUE(!hv.empty(), "No HV have been set.");

    for(uint8_t n = 0; n < hv.size(); n++) {
        hv[n]->disable();
    }
}

std::pair<Buffer::SharedHandle, arrus::session::Metadata::SharedHandle>
Us4RImpl::upload(const Scheme &scheme) {
    auto &outputBufferSpec = scheme.getOutputBuffer();
    auto rxBufferNElements = scheme.getRxBufferSize();
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
    auto &seq = scheme.getTxRxSequence();

    auto [rxBuffer, fcm] = uploadSequence(seq, rxBufferNElements, seq.getNRepeats(), scheme.getWorkMode(),
                                          scheme.getDigitalDownConversion(), scheme.getConstants());

    prepareHostBuffer(hostBufferNElements, workMode, rxBuffer);
    // NOTE: starting from this point, rxBuffer is no longer a valid variable
    // Metadata
    arrus::session::MetadataBuilder metadataBuilder;
    metadataBuilder.add<FrameChannelMapping>("frameChannelMapping", std::move(fcm));
    this->currentScheme = scheme;
    return {this->buffer, metadataBuilder.buildPtr()};
}

void Us4RImpl::prepareHostBuffer(unsigned nElements, Scheme::WorkMode workMode, std::unique_ptr<Us4RBuffer> &rxBuffer,
                                 bool cleanupSequencer) {
    ARRUS_REQUIRES_TRUE(!rxBuffer->empty(), "Us4R Rx buffer cannot be empty.");

    // Calculate how much of the data each Us4OEM produces.
    auto &element = rxBuffer->getElement(0);
    // a vector, where value[i] contains a size that is produced by a single us4oem.
    std::vector<size_t> us4oemComponentSize(element.getNumberOfUs4oems(), 0);
    int i = 0;
    for (auto &component : element.getUs4oemComponents()) {
        us4oemComponentSize[i++] = component.getViewSize();
    }
    auto &shape = element.getShape();
    auto dataType = element.getDataType();
    // If the output buffer already exists - remove it.
    if (this->buffer) {
        // The buffer should be already unregistered (after stopping the device).
        this->buffer->shutdown();
        // We must be sure here, that there is no thread working on the us4rBuffer here.
        if (this->us4rBuffer) {
            unregisterOutputBuffer(cleanupSequencer);
            this->us4rBuffer.reset();
        }
        this->buffer.reset();
    }
    // Create output buffer.
    this->buffer =
        std::make_shared<Us4ROutputBuffer>(us4oemComponentSize, shape, dataType, nElements, stopOnOverflow);
    registerOutputBuffer(this->buffer.get(), rxBuffer, workMode);

    // Note: use only as a marker, that the upload was performed, and there is still some memory to unlock.
    this->us4rBuffer = std::move(rxBuffer);
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
        this->state = State::STOP_IN_PROGRESS;
        logger->log(LogSeverity::DEBUG, "Stopping system.");
        this->getDefaultComponent()->stop();
        try {
            for(auto &us4oem: us4oems) {
                us4oem->getIUs4oem()->WaitForPendingTransfers();
                us4oem->getIUs4oem()->WaitForPendingInterrupts();
            }
        }
        catch(const std::exception &e) {
            logger->log(LogSeverity::WARNING,
                        arrus::format("Error on waiting for pending interrupts and transfers: {}", e.what()));
        }
        // Here all us4R IRQ threads should not work anymore.
        // Cleanup.
        for (auto &us4oem : us4oems) {
            us4oem->getIUs4oem()->DisableInterrupts();
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
        if (this->buffer != nullptr) {
            this->buffer->shutdown();
            // We must be sure here, that there is no thread working on the us4rBuffer here.
            if (this->us4rBuffer) {
                unregisterOutputBuffer(false);
                this->us4rBuffer.reset();
            }
        }
        getDefaultLogger()->log(LogSeverity::INFO, "Connection to Us4R closed.");
    } catch(const std::exception &e) {
        std::cerr << "Exception while destroying handle to the Us4R device: " << e.what() << std::endl;
    }
}

std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
Us4RImpl::uploadSequence(const TxRxSequence &seq, uint16 bufferSize, uint16 batchSize,
                         arrus::ops::us4r::Scheme::WorkMode workMode,
                         const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                         const std::vector<framework::NdArray> &txDelayProfiles) {
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
    return getProbeImpl()->setTxRxSequence(actualSeq, seq.getTgcCurve(), bufferSize,
                                           batchSize, seq.getSri(), workMode, ddc, txDelayProfiles);
}

void Us4RImpl::trigger(bool sync, std::optional<long long> timeout) {
    this->getDefaultComponent()->syncTrigger();
    if(sync) {
        this->sync(timeout);
    }
}

void Us4RImpl::sync(std::optional<long long> timeout)  {
    for(auto &us4oem: us4oems) {
        us4oem->sync(timeout);
    }
}

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
    uint16 offset = 359;
    uint16 tgcT = 153;
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

uint16_t Us4RImpl::getAfe(uint8_t reg) {
    return us4oems[0]->getAfe(reg);
}

void Us4RImpl::setAfe(uint8_t reg, uint16_t val) {
    for (auto &us4oem : us4oems) {
        us4oem->setAfe(reg, val);
    }
}

void Us4RImpl::registerOutputBuffer(Us4ROutputBuffer *outputBuffer, const Us4RBuffer::Handle &us4rDDRBuffer,
                                            Scheme::WorkMode workMode) {
    Ordinal us4oemOrdinal = 0;

    if(transferRegistrar.size() < us4oems.size()) {
        transferRegistrar.resize(us4oems.size());
    }
    for(auto &us4oem: us4oems) {
        auto us4oemBuffer = us4rDDRBuffer->getUs4oemBuffer(us4oemOrdinal);
        this->registerOutputBuffer(outputBuffer, us4oemBuffer, us4oem.get(), workMode);
        ++us4oemOrdinal;
    }
}

/**
 * - This function assumes, that the size of output buffer (number of elements)
 *  is a multiple of number of us4oem elements.
 * - this function will not schedule data transfer when the us4oem element size is 0.
 */
void Us4RImpl::registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc,
                                    Us4OEMImplBase *us4oem, Scheme::WorkMode workMode) {
    auto us4oemOrdinal = us4oem->getDeviceId().getOrdinal();
    auto ius4oem = us4oem->getIUs4oem();
    const auto nElementsSrc = bufferSrc.getNumberOfElements();
    const size_t nElementsDst = bufferDst->getNumberOfElements();
    size_t elementSize = getUniqueUs4OEMBufferElementSize(bufferSrc);
    if (elementSize == 0) {
        return;
    }
    transferRegistrar[us4oemOrdinal] = std::make_shared<Us4OEMDataTransferRegistrar>(bufferDst, bufferSrc, us4oem);
    transferRegistrar[us4oemOrdinal]->registerTransfers();
    // Register buffer element release functions.
    bool isMaster = us4oem->getDeviceId().getOrdinal() == this->getMasterUs4oem()->getDeviceId().getOrdinal();
    size_t nRepeats = nElementsDst/nElementsSrc;
    uint16 startFiring = 0;
    for(size_t i = 0; i < bufferSrc.getNumberOfElements(); ++i) {
        auto &srcElement = bufferSrc.getElement(i);
        uint16 endFiring = srcElement.getFiring();
        for(size_t j = 0; j < nRepeats; ++j) {
            std::function<void()> releaseFunc = createReleaseCallback(workMode, startFiring, endFiring);
            bufferDst->registerReleaseFunction(j*nElementsSrc+i, releaseFunc);
        }
        startFiring = endFiring+1;
    }
    // Overflow handling
    ius4oem->RegisterReceiveOverflowCallback(createOnReceiveOverflowCallback(workMode, bufferDst, isMaster));
    ius4oem->RegisterTransferOverflowCallback(createOnTransferOverflowCallback(workMode, bufferDst, isMaster));
    // Work mode specific initialization
    if(workMode == ops::us4r::Scheme::WorkMode::SYNC) {
        ius4oem->EnableWaitOnReceiveOverflow();
        ius4oem->EnableWaitOnTransferOverflow();
    }
}

size_t Us4RImpl::getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const {
    std::unordered_set<size_t> sizes;
    for (auto &element: us4oemBuffer.getElements()) {
        sizes.insert(element.getViewSize());
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
            transferRegistrar[i].reset();
        }
    }
}

std::function<void()> Us4RImpl::createReleaseCallback(
    Scheme::WorkMode workMode, uint16 startFiring, uint16 endFiring) {

    switch(workMode) {
    case Scheme::WorkMode::HOST: // Automatically generate new trigger after releasing all elements.
        return [this, startFiring, endFiring]() {
          for(int i = (int)us4oems.size()-1; i >= 0; --i) {
              us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForReceive(startFiring, endFiring);
              us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
          }
          if(this->state != State::STOP_IN_PROGRESS && this->state != State::STOPPED) {
              getMasterUs4oem()->syncTrigger();
          }
        };
    case Scheme::WorkMode::ASYNC: // Trigger generator: us4R
    case Scheme::WorkMode::SYNC:  // Trigger generator: us4R
    case Scheme::WorkMode::MANUAL:// Trigger generator: external (e.g. user)
    case Scheme::WorkMode::MANUAL_OP:
        return [this, startFiring, endFiring]() {
          for(int i = (int)us4oems.size()-1; i >= 0; --i) {
              us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForReceive(startFiring, endFiring);
              us4oems[i]->getIUs4oem()->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
          }
        };
    default:
        throw ::arrus::IllegalArgumentException("Unsupported work mode.");
    }
}

std::function<void()> Us4RImpl::createOnReceiveOverflowCallback(
    Scheme::WorkMode workMode, Us4ROutputBuffer *outputBuffer, bool isMaster) {

    using namespace std::chrono_literals;
    switch(workMode) {
    case Scheme::WorkMode::SYNC:
        return  [this, outputBuffer, isMaster]() {
          try {
              this->logger->log(LogSeverity::WARNING, "Detected RX data overflow.");
              size_t nElements = outputBuffer->getNumberOfElements();
              // Wait for all elements to be released by the user.
              while(nElements != outputBuffer->getNumberOfElementsInState(framework::BufferElement::State::FREE)) {
                  std::this_thread::sleep_for(1ms);
                  if (this->state != State::STARTED) {
                      // Device is no longer running, exit gracefully.
                      return;
                  }
              }
              // Inform about free elements only once, in the master's callback.
              if(isMaster) {
                  for(int i = (int)us4oems.size()-1; i >= 0; --i) {
                      us4oems[i]->getIUs4oem()->SyncReceive();
                  }
              }
              outputBuffer->runOnOverflowCallback();
          } catch (const std::exception &e) {
              logger->log(LogSeverity::ERROR, format("RX overflow callback exception: ", e.what()));
          } catch (...) {
              logger->log(LogSeverity::ERROR, "RX overflow callback exception: unknown");
          }
        };
    case Scheme::WorkMode::ASYNC:
    case Scheme::WorkMode::HOST:
    case Scheme::WorkMode::MANUAL:
    case Scheme::WorkMode::MANUAL_OP:
        return [this, outputBuffer]() {
          try {
              if(outputBuffer->isStopOnOverflow()) {
                  this->logger->log(LogSeverity::ERROR, "Rx data overflow, stopping the device.");
                  this->getMasterUs4oem()->stop();
                  outputBuffer->markAsInvalid();
              } else {
                  this->logger->log(LogSeverity::WARNING, "Rx data overflow ...");
                  outputBuffer->runOnOverflowCallback();
              }
          } catch (const std::exception &e) {
              logger->log(LogSeverity::ERROR, format("RX overflow callback exception: ", e.what()));
          } catch (...) {
              logger->log(LogSeverity::ERROR, "RX overflow callback exception: unknown");
          }
        };
    default:
        throw ::arrus::IllegalArgumentException("Unsupported work mode.");
    }
}

std::function<void()> Us4RImpl::createOnTransferOverflowCallback(
    Scheme::WorkMode workMode, Us4ROutputBuffer *outputBuffer, bool isMaster) {

    using namespace std::chrono_literals;
    switch(workMode) {
    case Scheme::WorkMode::SYNC:
        return  [this, outputBuffer, isMaster]() {
          try {
              this->logger->log(LogSeverity::WARNING, "Detected host data overflow.");
              size_t nElements = outputBuffer->getNumberOfElements();
              // Wait for all elements to be released by the user.
              while(nElements != outputBuffer->getNumberOfElementsInState(framework::BufferElement::State::FREE)) {
                  std::this_thread::sleep_for(1ms);
                  if (this->state != State::STARTED) {
                      // Device is no longer running, exit gracefully.
                      return;
                  }
              }
              // Inform about free elements only once, in the master's callback.
              if(isMaster) {
                  for(int i = (int)us4oems.size()-1; i >= 0; --i) {
                      us4oems[i]->getIUs4oem()->SyncTransfer();
                  }
              }
              outputBuffer->runOnOverflowCallback();
          } catch (const std::exception &e) {
              logger->log(LogSeverity::ERROR, format("Host overflow callback exception: ", e.what()));
          } catch (...) {
              logger->log(LogSeverity::ERROR, "Host overflow callback exception: unknown");
          }
        };
    case Scheme::WorkMode::ASYNC:
    case Scheme::WorkMode::HOST:
    case Scheme::WorkMode::MANUAL:
    case Scheme::WorkMode::MANUAL_OP:
        return [this, outputBuffer]() {
          try {
              if(outputBuffer->isStopOnOverflow()) {
                  this->logger->log(LogSeverity::ERROR, "Host data overflow, stopping the device.");
                  this->getMasterUs4oem()->stop();
                  outputBuffer->markAsInvalid();
              } else {
                  outputBuffer->runOnOverflowCallback();
                  this->logger->log(LogSeverity::WARNING, "Host data overflow ...");
              }
          } catch (const std::exception &e) {
              logger->log(LogSeverity::ERROR, format("Host overflow callback exception: ", e.what()));
          } catch (...) {
              logger->log(LogSeverity::ERROR, "Host overflow callback exception: unknown");
          }
        };
    default:
        throw ::arrus::IllegalArgumentException("Unsupported work mode.");
    }
}

const char *Us4RImpl::getBackplaneSerialNumber() {
    if(!this->digitalBackplane.has_value()) {
        throw arrus::IllegalArgumentException("No backplane defined.");
    }
    return this->digitalBackplane->get()->getSerialNumber();
}

const char *Us4RImpl::getBackplaneRevision() {
    if(!this->digitalBackplane.has_value()) {
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
        this->us4oems[0]->getIUs4oem()->TriggerStop();
        try {
            for (auto &us4oem : us4oems) {
                us4oem->getIUs4oem()->SetTxDelays(value);
            }
        } catch (...) {
            // Try resume.
            this->us4oems[0]->getIUs4oem()->TriggerStart();
            throw;
        }
        // Everything OK, resume.
        this->us4oems[0]->getIUs4oem()->TriggerStart();
    }
}

std::pair<std::shared_ptr<Buffer>, std::shared_ptr<session::Metadata>>
Us4RImpl::setSubsequence(uint16_t start, uint16_t end, const std::optional<float> &sri) {
    if(!this->currentScheme.has_value()) {
        throw IllegalStateException("Please upload scheme before setting sub-sequence.");
    }
    const auto &s = this->currentScheme.value();
    const auto &seq = s.getTxRxSequence();
    const auto currentSequenceSize = static_cast<uint16_t>(seq.getOps().size());
    if(end >= currentSequenceSize) {
        throw IllegalArgumentException(format("The new sub-sequence [{}, {}] is outside of the scope of the currently "
                                       "uploaded sequence: [0, {})", start, end, currentSequenceSize));
    }
    auto [rxBuffer, fcm] = this->getProbeImpl()->setSubsequence(start, end, sri);
    prepareHostBuffer(s.getOutputBuffer().getNumberOfElements(), s.getWorkMode(), rxBuffer, true);
    arrus::session::MetadataBuilder metadataBuilder;
    metadataBuilder.add<FrameChannelMapping>("frameChannelMapping", std::move(fcm));
    return {this->buffer, metadataBuilder.buildPtr()};
}

void Us4RImpl::setMaximumPulseLength(std::optional<float> maxLength) {
    for(auto &us4oem: us4oems) {
        us4oem->setMaximumPulseLength(maxLength);
    }
}


}// namespace arrus::devices
