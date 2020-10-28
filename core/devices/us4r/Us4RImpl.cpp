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

void Us4RImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                               const ::arrus::ops::us4r::TGCCurve &tgcSamples) {
    getDefaultComponent()->setTxRxSequence(seq, tgcSamples);
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

std::tuple<
    FrameChannelMapping::Handle,
    HostBuffer::SharedHandle
>
Us4RImpl::upload(const ops::us4r::TxRxSequence &seq) {
    ARRUS_REQUIRES_EQUAL(
        getDefaultComponent(), probe.value().get(),
        ::arrus::IllegalArgumentException(
            "Currently TxRx sequence upload is available for system with probes only.")
    );
    std::vector<TxRxParameters> actualSeq;

    constexpr uint16_t N_ELEMENTS = 2;

    auto nus4oems = probeAdapter.value()->getNumberOfUs4OEMs();
    this->currentRxBuffer = std::make_unique<RxBuffer>(nus4oems, N_ELEMENTS);

    // Load n-times, each time for subsequent seq. element.

    auto nOps = seq.getOps().size();
    for(int i = 0; i < N_ELEMENTS; ++i) {

        size_t opIdx = 0;
        for(const auto&[tx, rx] : seq.getOps()) {
            std::optional<TxRxParameters::SequenceCallback> callback = nullptr;

            // Set checkpoint callback for the last tx/rx.
            if(opIdx == nOps - 1) {
                callback = [&, this, i](Ordinal us4oemOrdinal) {
                    logger->log(LogSeverity::DEBUG,
                                ::arrus::format("Notifying about new buffer element {}.", i));
                    this->currentRxBuffer->notify(us4oemOrdinal, i);
                    logger->log(LogSeverity::DEBUG,
                                ::arrus::format("Reserving element {}.", i));
                    bool isReservationPossible =
                        this->currentRxBuffer->reserveElement((i + 1) % N_ELEMENTS);
                    logger->log(LogSeverity::DEBUG,
                                ::arrus::format("Element {} reserved.", i));
                    return isReservationPossible;
                };
            }
            actualSeq.emplace_back(
                tx.getAperture(),
                tx.getDelays(),
                tx.getExcitation(),
                rx.getAperture(),
                rx.getSampleRange(),
                rx.getDownsamplingFactor(),
                seq.getPri(),
                rx.getPadding(),
                callback
            );
            ++opIdx;
        }
    }
    auto[fcm, transfers] = getDefaultComponent()->setTxRxSequence(actualSeq, seq.getTgcCurve());

    // transfers[i][j] = transfer to perform
    // where i is the section (buffer element), j is the us4oem (a part of the buffer element)
    // Currently we assume that each buffer element has the same size.
    size_t bufferElementSize = countBufferElementSize(transfers);

    this->hostBuffer = std::make_shared<Us4RHostBuffer>(
        bufferElementSize, N_ELEMENTS);
    this->dataCarrier = std::make_unique<HostBufferWorker>(

        this->currentRxBuffer, this->hostBuffer, transfers);
    return std::make_tuple(std::move(fcm), hostBuffer);
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
    this->getDefaultComponent()->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    if(this->currentRxBuffer) {
        this->currentRxBuffer->shutdown();
    }
    if(this->hostBuffer) {
        this->hostBuffer->shutdown();
    }
    this->state = State::STOPPED;
}

Us4RImpl::~Us4RImpl() {
    getDefaultLogger()->log(LogSeverity::DEBUG, "Destroying Us4R instance");
    // TODO mutex
    if(this->state == State::STARTED) {
        this->stopDevice();
    }
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
        throw ArrusException("A buffer elements with different sizes.");
    }
    return *std::begin(transferSizes);
}




}