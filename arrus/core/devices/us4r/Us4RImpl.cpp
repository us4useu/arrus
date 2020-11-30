#include "Us4RImpl.h"

#include <memory>
#include <chrono>
#include <thread>

namespace arrus::devices {

using ::arrus::ops::us4r::TxRxSequence;
using ::arrus::ops::us4r::Tx;
using ::arrus::ops::us4r::Rx;
using ::arrus::ops::us4r::Pulse;

UltrasoundDevice *Us4RImpl::getDefaultComponent() {
    // NOTE! The implementation of this function determines
    // validation behaviour of SetVoltage function.
    // The safest option is to prefer using Probe only,
    // with an option to choose us4oem
    // (but the user has to specify it explicitly in settings).
    // Currently there should be no option to set TxRxSequence
    // on an adapter directly.
    if(probe.has_value()) {
        return probe.value().get();
    } else {
        return us4oems[0].get();
    }
}

Us4RImpl::Us4RImpl(const DeviceId &id,
                   Us4RImpl::Us4OEMs us4oems,
                   ProbeAdapterImplBase::Handle &probeAdapter,
                   ProbeImplBase::Handle &probe,
                   std::optional<HV256Impl::Handle> hv)
    : Us4R(id), logger{getLoggerFactory()->getLogger()},
      us4oems(std::move(us4oems)),
      probeAdapter(std::move(probeAdapter)),
      probe(std::move(probe)),
      hv(std::move(hv)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());

}

void Us4RImpl::setVoltage(Voltage voltage) {
    logger->log(LogSeverity::INFO,
                ::arrus::format("Setting voltage {}", voltage));
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    // Validate.
    auto *device = getDefaultComponent();
    auto voltageRange = device->getAcceptedVoltageRange();

    auto minVoltage = voltageRange.start();
    auto maxVoltage = voltageRange.end();

    if(voltage < minVoltage || voltage > maxVoltage) {
        throw IllegalArgumentException(
            ::arrus::format("Unaccepted voltage '{}', "
                            "should be in range: [{}, {}]",
                            voltage, minVoltage, maxVoltage));
    }
    hv.value()->setVoltage(voltage);
}

void Us4RImpl::disableHV() {
    logger->log(LogSeverity::INFO, "Disabling HV");
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    hv.value()->disable();
}

std::pair<HostBuffer::SharedHandle, FrameChannelMapping::SharedHandle>
Us4RImpl::upload(const ops::us4r::TxRxSequence &seq,
                 unsigned short rxBufferNElements, unsigned short hostBufferNElements) {

    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        ::arrus::IllegalArgumentException(
            "Currently TxRx sequence upload is available for system with probes only."));

    if((hostBufferNElements % rxBufferNElements) != 0) {
        throw ::arrus::IllegalArgumentException(
            "The size of the host buffer must be a multiple of the size "
            "of the rx buffer.");
    }
    std::unique_lock<std::mutex> guard(deviceStateMutex);

    if(this->state == State::STARTED) {
        throw ::arrus::IllegalStateException(
            "The device is running, uploading sequence is forbidden.");
    }

    auto[buffer, fcm] = uploadSequence(seq, rxBufferNElements, 1);

    ARRUS_REQUIRES_TRUE(!buffer->empty(), "Us4R Rx buffer cannot be empty.");

    auto &element = buffer->getElement(0);
    std::vector<size_t> elementPartSizes(element.getNumberOfUs4oems(), 0);
    std::transform(
        std::begin(element.getUs4oemElements()),
        std::end(element.getUs4oemElements()),
        std::begin(elementPartSizes),
        [](DataTransfer& transfer) {
            return transfer.getSize();
        });
    this->buffer = std::make_shared<Us4ROutputBuffer>(
        elementPartSizes, hostBufferNElements);
    getProbeImpl()->registerOutputBuffer(this->buffer.get(), buffer);
    return {this->buffer, std::move(fcm)};
}

void Us4RImpl::start() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    logger->log(LogSeverity::INFO, "Starting us4r.");
    if(this->buffer == nullptr) {
        throw ::arrus::IllegalArgumentException("Call upload function first.");
    }
    if(this->state == State::STARTED) {
        throw ::arrus::IllegalStateException("Device is already running.");
    }
    this->getDefaultComponent()->start();
    this->state = State::STARTED;
}

void Us4RImpl::stop() {
    this->stopDevice();
}

void Us4RImpl::stopDevice() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if(this->state != State::STARTED) {
        logger->log(LogSeverity::INFO, "Device Us4R is already stopped.");
    }
    else {
        logger->log(LogSeverity::DEBUG, "Stopping system.");
        this->getDefaultComponent()->stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }
    if(this->buffer != nullptr) {
        this->buffer->shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        this->buffer.reset();
    }
    this->state = State::STOPPED;
}

Us4RImpl::~Us4RImpl() {
    getDefaultLogger()->log(LogSeverity::DEBUG,
                            "Closing connection with Us4R.");
    this->stopDevice();
    getDefaultLogger()->log(LogSeverity::INFO, "Connection to Us4R closed.");
}

std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
Us4RImpl::uploadSequence(const ops::us4r::TxRxSequence &seq,
                         uint16_t rxBufferSize, uint16_t rxBatchSize) {
    std::vector<TxRxParameters> actualSeq;
    auto nOps = seq.getOps().size();
    // Convert to intermediate representation (TxRxParameters).
    size_t opIdx = 0;
    for(const auto &txrx : seq.getOps()) {
        auto &tx = txrx.getTx();
        auto &rx = txrx.getRx();

        Interval<uint32> sampleRange(rx.getSampleRange().first, rx.getSampleRange().second);
        Tuple<ChannelIdx> padding({rx.getPadding().first, rx.getPadding().second});

        actualSeq.push_back(
            TxRxParameters(
                tx.getAperture(),
                tx.getDelays(),
                tx.getExcitation(),
                rx.getAperture(),
                sampleRange,
                rx.getDownsamplingFactor(),
                txrx.getPri(),
                padding)
        );
        ++opIdx;
    }
    return getProbeImpl()->setTxRxSequence(actualSeq, seq.getTgcCurve(), rxBufferSize, rxBatchSize, seq.getSri());

}

void Us4RImpl::syncTrigger() {
    this->getDefaultComponent()->syncTrigger();
}

void Us4RImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints) {
    this->getDefaultComponent()->setTgcCurve(tgcCurvePoints);
}



}