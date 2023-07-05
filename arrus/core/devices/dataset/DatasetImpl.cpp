#include "DatasetImpl.h"
#include <cmath>
#include "arrus/core/common/logging.h"
#include "arrus/core/common/collections.h"
#include "arrus/common/format.h"

namespace arrus::devices {

using namespace arrus::framework;
using namespace arrus::session;

DatasetImpl::DatasetImpl(const DeviceId &id, std::string filepath, size_t datasetSize, ProbeModel probeModel)
    : Ultrasound(id), logger{getLoggerFactory()->getLogger()},
      filepath(std::move(filepath)),
      datasetSize(datasetSize),
      probeModel(std::move(probeModel)) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    logger->log(LogSeverity::INFO, ::arrus::format("Simulated mode, dataset: {}", filepath));
}

std::pair<Buffer::SharedHandle, Metadata::SharedHandle>
DatasetImpl::upload(const ops::us4r::Scheme &scheme) {
    this->currentScheme = scheme;
    auto &seq = this->currentScheme->getTxRxSequence();

    // Determine dimensions of each frame.
    // NOTE: assuming that each RX has the same number of channels and samples.
    size_t nTx = seq.getOps().size();
    auto [startSample, stopSample] = seq.getOps().at(0).getRx().getSampleRange();
    size_t nSamples = stopSample-startSample;
    auto &rxAperture = seq.getOps()[0].getRx().getAperture();
    size_t nRx = std::reduce(std::begin(rxAperture), std::end(rxAperture));
    size_t nValues = this->currentScheme->getDigitalDownConversion().has_value() ? 2 : 1; // I/Q or raw data.

    this->buffer = std::make_shared<DatasetBuffer>(this->datasetSize, nTx, nSamples, nRx, nValues);

    // Metadata
    MetadataBuilder metadataBuilder;

    return std::make_pair(this->buffer, metadataBuilder.buildPtr());
}

void DatasetImpl::start() {
    // TODO mutex
    this->state = State::STARTED;
    this->producerThread = std::thread(&DatasetImpl::producer, this);
}

void DatasetImpl::stop() {
    // TODO mutex
    this->state = State::STOPPED;
    this->producerThread.join();
}

void DatasetImpl::producer() {
    size_t i = 0;
    logger->log(LogSeverity::INFO, "Starting dataset producer.");
    while(this->state == State::STARTED) {
        this->buffer.callOnNewDataCallback(i);
        i = (i+1) % this->datasetSize;
    }
    logger->log(LogSeverity::INFO, "Dataset producer stopped.");
}

std::vector<float> DatasetImpl::getTgcCurvePoints(float maxT) const {
    // TODO implement
    float nominalFs = getSamplingFrequency();
    uint16 offset = 359;
    uint16 tgcT = 153;
    uint16 maxNSamples = int16(roundf(maxT*nominalFs));
    // Note: the last TGC sample should be applied before the reception ends.
    // This is to avoid using the same TGC curve between triggers.
    auto values = ::arrus::getRange<uint16>(offset, maxNSamples, tgcT);
    values.push_back(maxNSamples);
    std::vector<float> time;
    for(auto v: values) {
        time.push_back(v/nominalFs);
    }
    return time;
}

void DatasetImpl::setVoltage(Voltage voltage) {/*NOP*/}
unsigned char DatasetImpl::getVoltage() { return 5; }
float DatasetImpl::getMeasuredPVoltage() { return 5; }
float DatasetImpl::getMeasuredMVoltage() { return 5; }
void DatasetImpl::disableHV() {/*NOP*/}
void DatasetImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints) {/*NOP*/}
void DatasetImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) {/*NOP*/}
void DatasetImpl::setTgcCurve(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic) {/*NOP*/}
void DatasetImpl::setPgaGain(uint16 value) {/*NOP*/}
uint16 DatasetImpl::getPgaGain() { return 0; }
void DatasetImpl::setLnaGain(uint16 value) {}
uint16 DatasetImpl::getLnaGain() { return 0; }
void DatasetImpl::setLpfCutoff(uint32 value) {}
void DatasetImpl::setDtgcAttenuation(std::optional<uint16> value) {}
void DatasetImpl::setActiveTermination(std::optional<uint16> value) {}

float DatasetImpl::getSamplingFrequency() const { return 65e6; }
float DatasetImpl::getCurrentSamplingFrequency() const { return 65e6; }
void DatasetImpl::setHpfCornerFrequency(uint32_t frequency) {}
void DatasetImpl::disableHpf() {}

size_t DatasetBuffer::getNumberOfElements() const { return 0; }
std::shared_ptr<arrus::framework::BufferElement> DatasetBuffer::getElement(size_t i) {
    return std::shared_ptr<BufferElement>();
}
size_t DatasetBuffer::getElementSize() const { return 0; }
size_t DatasetBuffer::getNumberOfElementsInState(framework::BufferElement::State state) const { return 0; }
DatasetBuffer::DatasetBuffer(std::unique_ptr<int16> data, size_t size) {}

void DatasetBuffer::registerOnNewDataCallback(framework::OnNewDataCallback &callback) {}
void DatasetBuffer::registerOnOverflowCallback(framework::OnOverflowCallback &callback) {}
void DatasetBuffer::registerShutdownCallback(framework::OnShutdownCallback &callback) {}

size_t DatasetBuffer::getNumberOfElements() const { return 0; }
std::shared_ptr<BufferElement> DatasetBuffer::getElement(size_t i) { return std::shared_ptr<BufferElement>(); }
size_t DatasetBuffer::getElementSize() const { return 0; }
size_t DatasetBuffer::getNumberOfElementsInState(framework::BufferElement::State state) const { return 0; }

}// namespace arrus::devices
