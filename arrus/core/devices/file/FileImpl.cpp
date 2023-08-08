#include "FileImpl.h"
#include "arrus/common/format.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/logging.h"
#include <cmath>

namespace arrus::devices {

using namespace arrus::framework;
using namespace arrus::session;

FileImpl::FileImpl(const DeviceId &id, const std::string &filepath, size_t datasetSize, ProbeModel probeModel)
    : Ultrasound(id), logger{getLoggerFactory()->getLogger()},
      datasetSize(datasetSize),
      probeModel(std::move(probeModel)) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    this->logger->log(LogSeverity::INFO, ::arrus::format("Simulated mode, dataset: {}", filepath));
    this->dataset = readDataset(filepath);
}

std::pair<Buffer::SharedHandle, Metadata::SharedHandle> FileImpl::upload(const ops::us4r::Scheme &scheme) {
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
    this->frameShape = NdArray::Shape{nTx, nSamples, nRx, nValues};
    this->buffer = std::make_shared<DatasetBuffer>(this->datasetSize, this->frameShape);
    // Metadata
    MetadataBuilder metadataBuilder;
    return std::make_pair(this->buffer, metadataBuilder.buildPtr());
}

void FileImpl::start() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if (this->state == State::STARTED) {
        logger->log(LogSeverity::INFO, "Already started.");
    } else {
        this->state = State::STARTED;
        this->producerThread = std::thread(&FileImpl::producer, this);
    }
}

void FileImpl::stop() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if(this->state == State::STOPPED) {
        logger->log(LogSeverity::INFO, "Already stopped.");
    }
    else {
        this->state = State::STOPPED;
        this->producerThread.join();
    }
}

void FileImpl::producer() {
    size_t i = 0;
    logger->log(LogSeverity::INFO, "Starting producer.");
    while(this->state == State::STARTED) {
        auto element = this->buffer.acquire();
        this->buffer.callOnNewDataCallback(i);
        i = (i+1) % this->datasetSize;
    }
    logger->log(LogSeverity::INFO, "Dataset producer stopped.");
}

std::vector<float> FileImpl::getTgcCurvePoints(float maxT) const {
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

void FileImpl::setVoltage(Voltage voltage) {/*NOP*/}
unsigned char FileImpl::getVoltage() { return 5; }
float FileImpl::getMeasuredPVoltage() { return 5; }
float FileImpl::getMeasuredMVoltage() { return 5; }
void FileImpl::disableHV() {/*NOP*/}
void FileImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints) {/*NOP*/}
void FileImpl::setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) {/*NOP*/}
void FileImpl::setTgcCurve(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic) {/*NOP*/}
void FileImpl::setPgaGain(uint16 value) {/*NOP*/}
uint16 FileImpl::getPgaGain() { return 0; }
void FileImpl::setLnaGain(uint16 value) {}
uint16 FileImpl::getLnaGain() { return 0; }
void FileImpl::setLpfCutoff(uint32 value) {}
void FileImpl::setDtgcAttenuation(std::optional<uint16> value) {}
void FileImpl::setActiveTermination(std::optional<uint16> value) {}

float FileImpl::getSamplingFrequency() const { return 65e6; }
float FileImpl::getCurrentSamplingFrequency() const { return 65e6; }
void FileImpl::setHpfCornerFrequency(uint32_t frequency) {}
void FileImpl::disableHpf() {}

}// namespace arrus::devices
