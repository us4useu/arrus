#include "Us4RImpl.h"

#include <memory>
#include <chrono>
#include <thread>

namespace arrus::devices {

using ::arrus::ops::us4r::TxRxSequence;
using ::arrus::ops::us4r::Tx;
using ::arrus::ops::us4r::Rx;
using ::arrus::ops::us4r::Pulse;

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

void Us4RImpl::disableHV() {
    logger->log(LogSeverity::DEBUG, "Disabling HV");
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    hv.value()->disable();
}

std::pair<
    FrameChannelMapping::SharedHandle,
    HostBuffer::SharedHandle
>
Us4RImpl::upload(const ops::us4r::TxRxSequence &seq) {
    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        ::arrus::IllegalArgumentException(
            "Currently TxRx sequence upload is available for system with probes only.")
    );
    std::vector<TxRxParameters> actualSeq;

    constexpr uint16_t BUFFER_SIZE = 2;
    auto nus4oems = (Ordinal)1;// probeAdapter.value()->getNumberOfUs4OEMs();
    this->currentRxBuffer = std::make_unique<RxBuffer>(nus4oems, BUFFER_SIZE);

    auto nOps = seq.getOps().size();
    size_t opIdx = 0;
    for(const auto& txrx : seq.getOps()) {
        auto &tx = txrx.getTx();
        auto &rx = txrx.getRx();
        std::optional<TxRxParameters::SequenceCallback> callback = std::nullopt;

        // Set checkpoint callback for the last tx/rx.
        if(opIdx == nOps - 1) {
            callback = [this, BUFFER_SIZE] (Ordinal us4oemOrdinal, uint16 i) {
                logger->log(LogSeverity::DEBUG,
                            ::arrus::format("Notify rx {}:{}.", us4oemOrdinal, i));
                bool canContinue = this->currentRxBuffer->notify(us4oemOrdinal, i);
                if(!canContinue) {
                    return false;
                }
                logger->log(LogSeverity::DEBUG,
                            ::arrus::format("Reserve rx {}:{}.", us4oemOrdinal, (i + 1) % BUFFER_SIZE));
                bool isReservationPossible =
                    this->currentRxBuffer->reserveElement((i + 1) % BUFFER_SIZE);
                logger->log(LogSeverity::DEBUG,
                            ::arrus::format("Rx Reserved {}:{}.", us4oemOrdinal, (i + 1) % BUFFER_SIZE));
                return isReservationPossible;
            };
        }

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
                seq.getPri(),
                padding,
                callback != std::nullopt,
                callback
            )
        );
        ++opIdx;
    }
    auto[fcm, transfers, nTriggers] = getDefaultComponent()->setTxRxSequence(
        actualSeq, seq.getTgcCurve(), BUFFER_SIZE);

    // transfers[i][j] = transfer to perform
    // where i is the section (buffer element), j is the us4oem (a part of the buffer element)
    // Currently we assume that each buffer element has the same size.
    size_t bufferElementSize = countBufferElementSize(transfers);

    this->hostBuffer = std::make_shared<Us4RHostBuffer>(bufferElementSize, BUFFER_SIZE);


    // Rx DMA timeout - to avoid situation, where rx irq is missing.
    // 1.5 - sleep time multiplier
    auto timeout = (long long) ((float)nTriggers*seq.getPri()*1e6*1.5);
    this->dataCarrier = std::make_unique<HostBufferWorker>(
        this->currentRxBuffer, this->hostBuffer, transfers, timeout);
    return std::make_pair(std::move(fcm), hostBuffer);
}

void Us4RImpl::start() {
    logger->log(LogSeverity::DEBUG, "Starting us4r.");
    if(this->dataCarrier == nullptr) {
        throw ::arrus::IllegalArgumentException("Call upload function first.");
    }
    this->dataCarrier->start();
    this->getDefaultComponent()->start();
    this->state = State::STARTED;
}

/**
 * When the device is stopped, the buffer is destroyed.
 */
void Us4RImpl::stop() {
    logger->log(LogSeverity::DEBUG, "Stopping us4r.");
    this->stopDevice();
}

void Us4RImpl::stopDevice() {
    if(this->state != State::STARTED) {
        throw IllegalArgumentException("Device is not running.");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    this->getDefaultComponent()->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    if(this->currentRxBuffer) {
        this->currentRxBuffer->shutdown();
    }
    if(this->hostBuffer) {
        this->hostBuffer->shutdown();
    }
    this->state = State::STOPPED;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
}

Us4RImpl::~Us4RImpl() {
    getDefaultLogger()->log(LogSeverity::DEBUG, "Closing connection with Us4R.");
    if(this->state == State::STARTED) {
        this->stopDevice();
        this->dataCarrier->join();
    }
    getDefaultLogger()->log(LogSeverity::DEBUG, "Connection to Us4R closed.");
}

size_t Us4RImpl::countBufferElementSize(const std::vector<std::vector<DataTransfer>> &transfers) {
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


}