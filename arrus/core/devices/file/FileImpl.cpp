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

std::vector<FileImpl::Frame> FileImpl::readDataset(const std::string &filepath) {
    std::ifstream file{filepath, std::ios::in | std::ios::binary};
    std::istreambuf_iterator<char> start(file), end;
    std::vector<char> bytes(start, end);
    if(bytes.empty()) {
        throw ArrusException("Empty input file. Is your input file correct?");
    }
    if(bytes.size() % sizeof(int16_t) != 0) {
        throw ArrusException("Invalid input data size: the number of read bytes is not divisible by 2 (int16_t). "
                             "Is your input file correct?");
    }
    std::vector<int16_t> all(bytes.size() / sizeof(int16_t));
    if(all.size() % settings.getNFrames() != 0) {
        throw ArrusException(format(
            "Invalid input data size: the number of int16_t values {} is not divisible by {}. "
            "(the number of declared frames). Is your input file correct?",
            all.size(), settings.getNFrames()));
    }
    size_t frameSize = all.size() / settings.getNFrames();
    std::vector<Frame> result;
    for(size_t i = 0; i < settings.getNFrames(); ++i) {
        Frame frame(std::begin(all)+i*frameSize, std::begin(all)+(i+1)*frameSize);
        result.push_back(std::move(frame));
    }
    return result;
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
    // Check if the frame size from the dataset corresponds corresponds to the given frame shape.
    if(this->frameShape.product() != dataset.at(0).size()) {
        throw ArrusException(
            format("The provided sequence (output dimensions: nTx: {}, nSamples: {}, nRx: {}, nComponents: {})) "
                   "does not correspond to the data from the file (number of int16_t values: {}). "
                   "Please make sure you are uploading the correct sequence.",
                   nTx, nSamples, nRx, nValues, dataset.at(0).size()));
    }

    // Determine current sampling frequency
    if(this->currentScheme->getDigitalDownConversion().has_value()) {
        auto dec = this->currentScheme->getDigitalDownConversion().value().getDecimationFactor();
        this->currentFs = this->getSamplingFrequency()/dec;
    }
    else {
        auto dec = seq.getOps().at(0).getRx().getDownsamplingFactor();
        this->currentFs = this->getSamplingFrequency()/dec;
    }
    this->buffer = std::make_shared<FileBuffer>(this->settings.getNFrames(), this->frameShape);
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
        this->consumerThread = std::thread(&FileImpl::consumer, this);
    }
}

void FileImpl::stop() {
    std::unique_lock<std::mutex> guard(deviceStateMutex);
    if(this->state == State::STOPPED) {
        logger->log(LogSeverity::INFO, "Already stopped.");
    }
    else {
        this->state = State::STOPPED;
        this->buffer->close();
        guard.unlock();
        this->producerThread.join();
        this->consumerThread.join();
    }
}

void FileImpl::producer() {
    size_t elementNr = 0;
    size_t frameNr = 0;
    logger->log(LogSeverity::INFO, "Starting producer.");
    while(this->state == State::STARTED) {
        bool cont = buffer->write(elementNr, [this, &frameNr] (const framework::BufferElement::SharedHandle &element) {
            auto &frame = this->dataset.at(frameNr);
            std::memcpy(element->getData().get<int16_t>(), frame.data(), frame.size()*sizeof(int16_t));
        });
        if(!cont) {
            break;
        }
        elementNr = (elementNr+1) % buffer->getNumberOfElements();
        frameNr = (frameNr+1) % dataset.size();
    }
    logger->log(LogSeverity::INFO, "File producer stopped.");
}

void FileImpl::consumer() {
    size_t i = 0;
    logger->log(LogSeverity::INFO, "Starting consumer.");
    while(this->state == State::STARTED) {
        bool cont = buffer->read(i, [this] (const framework::BufferElement::SharedHandle &element) {
            this->buffer->getOnNewDataCallback()(element);
        });
        if(!cont) {
            break;
        }
    }
    logger->log(LogSeverity::INFO, "File consumer stopped.");
}

void FileImpl::trigger() {
    throw std::runtime_error("File::trigger: NYI");
}

float FileImpl::getSamplingFrequency() const { return 65e6; }
float FileImpl::getCurrentSamplingFrequency() const { return this->currentFs; }

}// namespace arrus::devices
