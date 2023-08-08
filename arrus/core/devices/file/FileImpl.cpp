#include "FileImpl.h"
#include "arrus/common/format.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/logging.h"
#include <cmath>

namespace arrus::devices {

using namespace arrus::framework;
using namespace arrus::session;

FileImpl::FileImpl(const DeviceId &id, const FileSettings &settings)
    : File(id), logger{getLoggerFactory()->getLogger()}, settings(settings) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    this->logger->log(LogSeverity::INFO, ::arrus::format("File device, path: {}", settings.getFilepath()));
    this->dataset = readDataset(settings.getFilepath());
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
    // Determine current sampling frequency
    if(this->currentScheme->getDigitalDownConversion().has_value()) {
        auto dec = this->currentScheme->getDigitalDownConversion().value().getDecimationFactor();
        this->currentFs = this->getSamplingFrequency()/dec;
    }
    else {
        auto dec = seq.getOps().at(0).getRx().getDownsamplingFactor();
        this->currentFs = this->getSamplingFrequency()/dec;
    }
    this->buffer = std::make_shared<DatasetBuffer>(this->settings.getNFrames(), this->frameShape);
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

void FileImpl::trigger() {
    throw std::runtime_error("File::trigger: NYI");
}

float FileImpl::getSamplingFrequency() const { return 65e6; }
float FileImpl::getCurrentSamplingFrequency() const { return this->currentFs; }

}// namespace arrus::devices
