#include "FileImpl.h"
#include "arrus/common/format.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/common/exceptions.h"
#include <cmath>
#include <utility>
#include <ctime>

namespace arrus::devices {

using namespace arrus::framework;
using namespace arrus::session;

FileImpl::FileImpl(const DeviceId &id, const FileSettings &settings)
    : File(id), logger{getLoggerFactory()->getLogger()}, settings(settings) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
    this->logger->log(LogSeverity::INFO, ::arrus::format("File device, path: {}", settings.getFilepath()));
    this->dataset = readDataset(settings.getFilepath());
    this->probe = std::make_unique<FileProbe>(id, settings.getProbeModel());
}

std::vector<FileImpl::Frame> FileImpl::readDataset(const std::string &filepath) {
    logger->log(LogSeverity::INFO, "Reading input dataset...");
    std::ifstream file{filepath, std::ios::in | std::ios::binary};
    // Read input file size.
    file.unsetf(std::ios::skipws);
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    logger->log(LogSeverity::INFO, format("Input file size: {} MiB", float(fileSize)/(1<<20)));
    if(fileSize == 0) {
        throw ArrusException("Empty input file. Is your input file correct?");
    }
    if(fileSize % sizeof(int16_t) != 0) {
        throw ArrusException("Invalid input data size: the number of read bytes is not divisible by 2 (int16_t). "
                             "Is your input file correct?");
    }
    std::vector<int16_t> all(fileSize / sizeof(int16_t));
    if(all.size() % settings.getNFrames() != 0) {
        throw ArrusException(format(
            "Invalid input data size: the number of int16_t values {} is not divisible by {}. "
            "(the number of declared frames). Is your input file correct?",
            all.size(), settings.getNFrames()));
    }
    file.read((char*)all.data(), fileSize);
    size_t frameSize = all.size()/settings.getNFrames();
    std::vector<Frame> result;
    for(size_t i = 0; i < settings.getNFrames(); ++i) {
        Frame frame(std::begin(all)+i*frameSize, std::begin(all)+(i+1)*frameSize);
        result.push_back(std::move(frame));
    }
    logger->log(LogSeverity::INFO, "Data ready.");
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
    size_t nRx = std::accumulate(std::begin(rxAperture), std::end(rxAperture), 0);
    nRx += seq.getOps()[0].getRx().getPadding().first;
    nRx += seq.getOps()[0].getRx().getPadding().second;
    size_t nValues = this->currentScheme->getDigitalDownConversion().has_value() ? 2 : 1; // I/Q or raw data.
    this->frameShape = NdArray::Shape{1, nTx, nRx, nSamples, nValues};
    this->txBegin = 0;
    this->txEnd = (int)nTx;
    // Check if the frame size from the dataset corresponds corresponds to the given frame shape.
    if(this->frameShape.product() != dataset.at(0).size()) {
        throw ArrusException(
            format("The provided sequence (output dimensions: nTx: {}, nRx: {}, nSamples: {}, nComponents: {})) "
                   "does not correspond to the data from the file (number of int16_t values: {}). "
                   "Please make sure you are uploading the correct sequence.",
                   nTx, nRx, nSamples, nValues, dataset.at(0).size()));
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
            auto fileBufferElement = std::dynamic_pointer_cast<FileBufferElement>(element);
            std::memcpy(fileBufferElement->getAllData().getInt16(), frame.data(), frame.size()*sizeof(int16_t));

            // Write us4R specific metadata
            // Zero metadata row: (32 int16)
            for(int i = 0; i < 32; ++i) {
                size_t offset = i*this->frameShape[3]*this->frameShape[4];
                *(element->getData().get<int16_t>()+offset) = 0;
            }
            size_t beginOffset = 8*this->frameShape[3]*this->frameShape[4];
            size_t endOffset = 9*this->frameShape[3]*this->frameShape[4];
            size_t timestampOffset = 4*this->frameShape[3]*this->frameShape[4];
            *(element->getData().get<int16_t>()+beginOffset) = (int16_t)this->txBegin;
            *(element->getData().get<int16_t>()+endOffset) = (int16_t)this->txEnd;
            *((int16_t*)(element->getData().get<int16_t>()+timestampOffset)) = (int16_t)std::time(nullptr);
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(50ms);
        });
        if(!cont) {
            break;
        }
        elementNr = (elementNr+1) % buffer->getNumberOfElements();
        frameNr = (frameNr+1) % dataset.size();
        // Update sequence parameters.
        {
            std::unique_lock<std::mutex> lock(parametersMutex);
            if(pendingSliceBegin.has_value() || pendingSliceEnd.has_value()) {
                int sliceBegin=0, sliceEnd=-1;

                if(pendingSliceBegin.has_value()) {
                    sliceBegin = pendingSliceBegin.value();
                    this->txBegin = sliceBegin;
                    pendingSliceBegin.reset();
                }
                if(pendingSliceEnd.has_value()) {
                    sliceEnd = pendingSliceEnd.value();
                    this->txEnd = sliceEnd;
                    pendingSliceEnd.reset();
                }
                buffer->slice(1, sliceBegin, sliceEnd);
            }
        }
    }
    logger->log(LogSeverity::INFO, "File producer stopped.");
}

void FileImpl::consumer() {
    size_t elementNr = 0;
    logger->log(LogSeverity::INFO, "Starting consumer.");
    while(this->state == State::STARTED) {
        bool cont = buffer->read(elementNr, [this] (const framework::BufferElement::SharedHandle &element) {
            this->buffer->getOnNewDataCallback()(element);
        });
        if(!cont) {
            break;
        }
        elementNr = (elementNr+1) % buffer->getNumberOfElements();
    }
    logger->log(LogSeverity::INFO, "File consumer stopped.");
}

void FileImpl::trigger() {
    throw std::runtime_error("File::trigger: NYI");
}

ProbeModel FileImpl::getProbeModel(Ordinal ordinal) {
    if(ordinal > 0) {
        throw IllegalArgumentException("Probe with ordinal > 0 is not available in the File device.");
    }
    return probe->getModel();
}

void FileImpl::setParameters(const Parameters &params) {
    std::unique_lock<std::mutex> lock(parametersMutex);
    for(auto &item: params.items()) {
        auto &key = item.first;
        auto value = item.second;
        if(key == "/sequence:0/begin") {
            if(value < 0) {
                throw ::arrus::IllegalArgumentException(::arrus::format("{} should be not less than 0", key));
            }
            pendingSliceBegin = value;
        }
        else if(key == "/sequence:0/end") {
            int currentNTx = (int)frameShape.get(1);
            if(value >= currentNTx) {
                throw ::arrus::IllegalArgumentException(::arrus::format("{} should be less than {}", key, currentNTx));
            }
            pendingSliceEnd = value;
        }
        else {
            throw ::arrus::IllegalArgumentException("Unsupported setting: " + key);
        }
    }
}

float FileImpl::getSamplingFrequency() const { return 65e6; }
float FileImpl::getCurrentSamplingFrequency() const { return this->currentFs; }

}// namespace arrus::devices
