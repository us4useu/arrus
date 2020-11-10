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
    logger->log(LogSeverity::DEBUG,
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
    logger->log(LogSeverity::DEBUG, "Disabling HV");
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    hv.value()->disable();
}

std::pair<
    FrameChannelMapping::SharedHandle,
    HostBuffer::SharedHandle
>
Us4RImpl::uploadAsync(const ops::us4r::TxRxSequence &seq,
                      unsigned short rxBufferSize,
                      unsigned short hostBufferSize,
                      float frameRepetitionInterval) {

    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        ::arrus::IllegalArgumentException(
            "Currently TxRx sequence upload is available for system with probes only.")
    );
    std::unique_lock<std::mutex> guard(deviceStateMutex);

    if(this->state == State::STARTED) {
        throw ::arrus::IllegalStateException(
            "The device is running, uploading sequence is forbidden.");
    }

    // TODO prepare host buffer data (check the implementation)
    // TODO ScheduleTransferRxBufferToHost callback: signal that the data is ready
    // if the callback is set, us4oemimpl.cpp should perform the approach as presented
    // in the
    // TODO register overflow callback
    auto[fcm, transfers, totalTime] = uploadSequence(
        seq, BUFFER_SIZE, true,
        std::optional<TxRxParameters::SequenceCallback>(),
        std::optional<TxRxParameters::SequenceCallback>());
}


std::pair<
    FrameChannelMapping::SharedHandle,
    HostBuffer::SharedHandle
>
Us4RImpl::uploadSync(const ops::us4r::TxRxSequence &seq) {
    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        ::arrus::IllegalArgumentException(
            "Currently TxRx sequence upload is available for system with probes only.")
    );
    std::unique_lock<std::mutex> guard(deviceStateMutex);

    if(this->state == State::STARTED) {
        throw ::arrus::IllegalStateException(
            "The device is already started, uploading sequence is forbidden.");
    }

    constexpr uint16_t BUFFER_SIZE = 2;
    auto nus4oems = (Ordinal) 1;// probeAdapter.value()->getNumberOfUs4OEMs();
    this->currentRxBuffer = std::make_unique<RxBuffer>(nus4oems, BUFFER_SIZE);
    this->watchdog = std::make_unique<Watchdog>();

    auto[fcm, transfers, totalTime] = uploadSequence(
        seq, BUFFER_SIZE, true,
        std::optional<TxRxParameters::SequenceCallback>(),
        std::optional<TxRxParameters::SequenceCallback>());

    // transfers[i][j] = transfer to perform
    // where i is the section (buffer element), j is the us4oem (a part of the buffer element)
    // Currently we assume that each buffer element has the same size.
    size_t bufferElementSize = countBufferElementSize(transfers);
    this->hostBuffer = std::make_shared<Us4RHostBuffer>(bufferElementSize, BUFFER_SIZE);
    // Rx DMA timeout - to avoid situation, where rx irq is missing.
    // 1.5 - sleep time multiplier

    logger->log(LogSeverity::DEBUG,
                ::arrus::format("Total PRI: {}", totalTime));
    auto timeout = (long long) (totalPri * 1e6 * 1.5);
    logger->log(LogSeverity::DEBUG,
                ::arrus::format("Host buffer worker timeout: {}", timeout));
    this->hostBufferWorker = std::make_unique<HostBufferWorker>(
        this->currentRxBuffer, this->hostBuffer, transfers);

    this->watchdog->setTimeout(timeout);
    this->watchdog->setCallback([this]() {
        return this->rxDmaCallback();
    });

    this->mode = SYNC;
    return std::make_pair(std::move(fcm), hostBuffer);
}

void Us4RImpl::start() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    logger->log(LogSeverity::DEBUG, "Starting us4r.");
    if(this->hostBufferWorker == nullptr) {
        throw ::arrus::IllegalArgumentException("Call upload function first.");
    }
    if(this->state == State::STARTED) {
        throw ::arrus::IllegalStateException("Device is already running.");
    }
    this->hostBufferWorker->start();
    this->watchdog->start();
    this->getDefaultComponent()->start();
    this->watchdog->notifyStart();
    this->state = State::STARTED;
}

/**
 * When the device is stopped, the buffer should be destroyed.
 */
void Us4RImpl::stop() {
    logger->log(LogSeverity::DEBUG, "Stopping us4r.");
    this->stopDevice();
}

void Us4RImpl::stopDevice(bool stopGently) {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if(this->state != State::STARTED) {
        if(stopGently) {
            logger->log(LogSeverity::INFO, "Device Us4R is already stopped.");
            return;
        } else {
            throw IllegalArgumentException("Device was not started.");
        }
    }
    logger->log(LogSeverity::DEBUG, "Queue shutdown.");
    if(this->currentRxBuffer) {
        this->currentRxBuffer->shutdown();
    }
    if(this->hostBuffer) {
        this->hostBuffer->shutdown();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    logger->log(LogSeverity::DEBUG, "Stopping watchdog.");
    this->watchdog->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    logger->log(LogSeverity::DEBUG, "Stopping system.");
    this->getDefaultComponent()->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    logger->log(LogSeverity::DEBUG, "Stopping host buffer worker.");
    this->hostBufferWorker->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    // Delete all data.
    this->currentRxBuffer.reset();
    this->hostBuffer.reset();
    this->hostBufferWorker.reset();

    this->state = State::STOPPED;
}

Us4RImpl::~Us4RImpl() {
    getDefaultLogger()->log(LogSeverity::DEBUG,
                            "Closing connection with Us4R.");
    this->stopDevice();
    getDefaultLogger()->log(LogSeverity::INFO, "Connection to Us4R closed.");
}

std::tuple<
    FrameChannelMapping::Handle,
    std::vector<std::vector<DataTransfer>>,
    uint16_t // ntriggers
>
Us4RImpl::uploadSequence(const ops::us4r::TxRxSequence &seq,
                         uint16_t nRepeats,
                         bool checkpoint,
                         std::optional<TxRxParameters::SequenceCallback> rxCallback,
                         std::optional<TxRxParameters::SequenceCallback> pcidmaCallback) {
    std::vector<TxRxParameters> actualSeq;
    auto nOps = seq.getOps().size();
    size_t opIdx = 0;
    for(const auto &txrx : seq.getOps()) {
        auto &tx = txrx.getTx();
        auto &rx = txrx.getRx();
        std::optional<TxRxParameters::SequenceCallback> callback = std::nullopt;


        Interval<uint32> sampleRange(rx.getSampleRange().first,
                                     rx.getSampleRange().second);
        Tuple<ChannelIdx> padding(
            {rx.getPadding().first, rx.getPadding().second});

        actualSeq.push_back(
            TxRxParameters(
                tx.getAperture(),
                tx.getDelays(),
                tx.getExcitation(),
                rx.getAperture(),
                sampleRange,
                rx.getDownsamplingFactor(),
                txrx.getPri(),
                padding,
                (opIdx == nOps - 1 && checkpoint),
                (opIdx == nOps - 1) ? rxCallback : std::nullopt,
                (opIdx == nOps - 1) ? pcidmaCallback : std::nullopt
            )
        );
        ++opIdx;
    }
    return getDefaultComponent()->setTxRxSequence(actualSeq, seq.getTgcCurve(), nRepeats);

}

size_t Us4RImpl::countBufferElementSize(
    const std::vector<std::vector<DataTransfer>> &transfers) {
    std::unordered_set<size_t> transferSizes;
    for(auto &bufferElement : transfers) {
        size_t size = 0;
        for(auto &us4oemTransfer : bufferElement) {
            size += us4oemTransfer.getSize();
        }
        transferSizes.insert(size);
    }
    if(transferSizes.size() > 1) {
        std::cout << ::arrus::toString(transferSizes) << std::endl;
        throw ArrusException("A buffer elements with different sizes.");
    }
    return *std::begin(transferSizes);
}

bool Us4RImpl::rxDmaCallback() {
    // Notify the new buffer element is available.
    bool canContinue = this->currentRxBuffer->notify(0);
    if(!canContinue) {
        return false;
    }
    // Reserve access to next element.
    bool isReservationPossible = this->currentRxBuffer->reserveElement(0);

    if(isReservationPossible) {
        // TODO access us4oem:0 directly (performance)?
        this->syncTrigger();
    }
    return isReservationPossible;
}

void Us4RImpl::syncTrigger() {
    this->getDefaultComponent()->syncTrigger();
}


}