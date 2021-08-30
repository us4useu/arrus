#include "Us4RImpl.h"

#include <memory>
#include <chrono>
#include <thread>

namespace arrus::devices {

using ::arrus::ops::us4r::TxRxSequence;
using ::arrus::ops::us4r::Tx;
using ::arrus::ops::us4r::Rx;
using ::arrus::ops::us4r::Pulse;
using ::arrus::framework::Buffer;
using ::arrus::framework::DataBufferSpec;

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

Us4RImpl::Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::optional<HighVoltageSupplier::Handle> hv)
    : Us4R(id), logger{getLoggerFactory()->getLogger()}, us4oems(std::move(us4oems)), hv(std::move(hv)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

Us4RImpl::Us4RImpl(const DeviceId &id, Us4RImpl::Us4OEMs us4oems, ProbeAdapterImplBase::Handle &probeAdapter,
                   ProbeImplBase::Handle &probe, std::optional<HighVoltageSupplier::Handle> hv)
    : Us4R(id), logger{getLoggerFactory()->getLogger()}, us4oems(std::move(us4oems)),
      probeAdapter(std::move(probeAdapter)), probe(std::move(probe)), hv(std::move(hv)) {

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
        throw IllegalArgumentException(::arrus::format("Unaccepted voltage '{}', should be in range: [{}, {}]",
                            voltage, minVoltage, maxVoltage));
    }
    hv.value()->setVoltage(voltage);
}

void Us4RImpl::disableHV() {
    logger->log(LogSeverity::INFO, "Disabling HV");
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    hv.value()->disable();
}

std::pair<Buffer::SharedHandle, FrameChannelMapping::SharedHandle>
Us4RImpl::upload(const ops::us4r::TxRxSequence &seq,
                 unsigned short rxBufferNElements,
                 const ::arrus::ops::us4r::Scheme::WorkMode &workMode,
                 const DataBufferSpec &outputBufferSpec) {

    unsigned hostBufferNElements = outputBufferSpec.getNumberOfElements();

    // Validate input parameters.
    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        ::arrus::IllegalArgumentException(
            "Currently TxRx sequence upload is available for system with probes only."));
    if ((hostBufferNElements % rxBufferNElements) != 0) {
        throw ::arrus::IllegalArgumentException(
            ::arrus::format(
                "The size of the host buffer {} must be equal or a multiple of the size of the rx buffer {}.",
                hostBufferNElements, rxBufferNElements));
    }
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state == State::STARTED) {
        throw ::arrus::IllegalStateException("The device is running, uploading sequence is forbidden.");
    }

    // Upload and register buffers.
    bool isTriggerSync = workMode == ::arrus::ops::us4r::Scheme::WorkMode::HOST;
    auto[rxBuffer, fcm] = uploadSequence(seq, rxBufferNElements, 1, isTriggerSync);
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
        this->buffer.reset();
    }
    // Create output buffer.
    this->buffer = std::make_shared<Us4ROutputBuffer>(us4oemComponentSize, shape, dataType, hostBufferNElements);
    getProbeImpl()->registerOutputBuffer(this->buffer.get(), rxBuffer, isTriggerSync);
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
    this->getDefaultComponent()->start();
    this->state = State::STARTED;
}

void Us4RImpl::stop() {
    this->stopDevice();
}

void Us4RImpl::stopDevice() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state != State::STARTED) {
        logger->log(LogSeverity::INFO, "Device Us4R is already stopped.");
    } else {
        logger->log(LogSeverity::DEBUG, "Stopping system.");
        this->getDefaultComponent()->stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }
    if (this->buffer != nullptr) {
        this->buffer->shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    this->state = State::STOPPED;
}

Us4RImpl::~Us4RImpl() {
    getDefaultLogger()->log(LogSeverity::DEBUG, "Closing connection with Us4R.");
    this->stopDevice();
    getDefaultLogger()->log(LogSeverity::INFO, "Connection to Us4R closed.");
}

std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
Us4RImpl::uploadSequence(const ops::us4r::TxRxSequence &seq, uint16_t rxBufferSize, uint16_t rxBatchSize,
                         bool triggerSync) {
    std::vector<TxRxParameters> actualSeq;
    // Convert to intermediate representation (TxRxParameters).
    size_t opIdx = 0;
    for (const auto &txrx : seq.getOps()) {
        auto &tx = txrx.getTx();
        auto &rx = txrx.getRx();

        Interval<uint32> sampleRange(rx.getSampleRange().first, rx.getSampleRange().second);
        Tuple<ChannelIdx> padding({rx.getPadding().first, rx.getPadding().second});

        actualSeq.push_back(
            TxRxParameters(tx.getAperture(), tx.getDelays(), tx.getExcitation(),rx.getAperture(),
                sampleRange, rx.getDownsamplingFactor(), txrx.getPri(), padding)
        );
        ++opIdx;
    }
    return getProbeImpl()->setTxRxSequence(actualSeq, seq.getTgcCurve(), rxBufferSize, rxBatchSize, seq.getSri(),
                                           triggerSync);
}

void Us4RImpl::syncTrigger() {
    this->getDefaultComponent()->syncTrigger();
}

// AFE parameters

void Us4RImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints) {
    setTgcCurve(tgcCurvePoints, true);
}

void Us4RImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) {
    // TODO AFE mutex
    // TODO tutaj najpierw powinna byc walidacja, potem ustawianie
    // Tylko informacja, ze poszlo cos nie tak w trakcie ustawiania parametru i parametry (wykrywac odpowiednia sytuacje)

    // validate
    // AFE mutex
    // turn off dtgc
    // set tgc curve
    if(! tgcCurvePoints.empty()) {
        setDtgcAttenuation(std::nullopt);
    }
    for(auto &us4oem: us4oems) {
        us4oem->setTgcCurve(tgcCurvePoints, applyCharacteristic);
    }
}

void Us4RImpl::setLpfCutoff(uint32 value) {
    for(auto &us4oem: us4oems) {
        us4oem->setLpfCutoff(value);
    }
}
void Us4RImpl::setActiveTermination(std::optional<uint16> value) {
    for(auto &us4oem: us4oems) {
        us4oem->setActiveTermination(value);
    }
}
void Us4RImpl::setDtgcAttenuation(std::optional<uint8> value) {
    if(value.has_value()) {

    }
    for(auto &us4oem: us4oems) {
        us4oem->setTgcCurve(tgcCurvePoints, applyCharacteristic);
    }
    for(auto &us4oem: us4oems) {
        us4oem->setDtgcAttenuation(std::nullopt);
    }

}
void Us4RImpl::setLnaGain(uint8 value) {

}
void Us4RImpl::setPgaGain(uint8 value) {

}

}