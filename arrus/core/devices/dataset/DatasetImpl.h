#ifndef ARRUS_CORE_DEVICES_DATASET_DATASETIMPL_H
#define ARRUS_CORE_DEVICES_DATASET_DATASETIMPL_H

#include <array>
#include <arrus/core/api/framework/NdArray.h>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <condition_variable>

#include "arrus/core/api/devices/Ultrasound.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/session/Metadata.h"

namespace arrus::devices {

class DatasetBufferElement: public arrus::framework::BufferElement {
public:
    DatasetBufferElement(size_t position, const arrus::framework::NdArray::Shape& shape) {
        this->size = shape.product(); // The number of int16 elements.
        this->data = new int16_t[size];
        this->ndarray = framework::NdArray{
            this->data, shape, arrus::framework::NdArray::DataType::INT16, DeviceId(DeviceType::CPU, 0)};
        this->position = position;
    }

    ~DatasetBufferElement() override = default;

    void acquire(const std::function<void(framework::BufferElement::BufferElement::SharedHandle)> &func) {
        std::unique_lock<std::mutex> lock{stateMutex};
        // Wait until the element is free.
        readyForWrite.wait(lock, [this](){return this->state == framework::BufferElement::State::FREE;});

    }

    void releaseForRead() {
        std::unique_lock<std::mutex> lock{stateMutex};
    }

    void release() override {
        // Release
        readyForWrite.notify_one();
    }
    framework::NdArray &getData() override { return ndarray; }
    size_t getSize() override { return size*sizeof(int16_t); }
    size_t getPosition() override { return position; }
    State getState() const override { return state; }
private:
    std::mutex stateMutex;
    std::condition_variable readyForWrite;
    int16_t *data{nullptr};
    size_t size;
    // NdArray: view of the above data pointer.
    arrus::framework::NdArray ndarray{
        data,
        arrus::framework::NdArray::Shape{},
        arrus::framework::NdArray::DataType::INT16,
        DeviceId{DeviceType::CPU, 0}
    };
    size_t position;
    State state{framework::BufferElement::State::FREE};
};

class DatasetBuffer: public arrus::framework::DataBuffer {
public:

    DatasetBuffer(size_t nElements, arrus::framework::NdArray::Shape shape){
        for(size_t i = 0; i < nElements; ++i) {
            elements.push_back(std::make_shared<DatasetBufferElement>(i, shape));
        }
    }

    ~DatasetBuffer() override = default;

    void registerOnNewDataCallback(framework::OnNewDataCallback &callback) override {
        this->onNewDataCallback = callback;
    }
    void registerOnOverflowCallback(framework::OnOverflowCallback&) override {/*Ignored*/}
    void registerShutdownCallback(framework::OnShutdownCallback&) override {/*Ignored*/}

    size_t getNumberOfElements() const override { return elements.size(); }

    std::shared_ptr<arrus::framework::BufferElement> getElement(size_t i) override {
        return elements.at(i);
    }
    size_t getElementSize() const override {
        if(elements.empty()) {
            throw std::runtime_error("The Dataset Buffer is empty.");
        }
        return elements.at(0)->getSize();
    }
    size_t getNumberOfElementsInState(framework::BufferElement::State state) const override {
        int result = 0;
        for(auto &e: elements) {
            if(e->getState() == state) {
                ++result;
            }
        }
        return result;
    }

private:
    std::vector<std::shared_ptr<DatasetBufferElement>> elements;
    arrus::framework::OnNewDataCallback onNewDataCallback;
};

class DatasetImpl : public Ultrasound {
public:
    enum class State { STARTED, STOPPED };

    ~DatasetImpl() override {}

    DatasetImpl(const DeviceId &id, std::string filepath, size_t datasetSize, ProbeModel probeModel);

    DatasetImpl(DatasetImpl const &) = delete;

    DatasetImpl(DatasetImpl const &&) = delete;

    std::pair<
        std::shared_ptr<arrus::framework::Buffer>,
        std::shared_ptr<arrus::session::Metadata>
    >
    upload(const ops::us4r::Scheme &scheme) override;
    void setVoltage(Voltage voltage) override;
    unsigned char getVoltage() override;
    float getMeasuredPVoltage() override;
    float getMeasuredMVoltage() override;
    void disableHV() override;
    void setTgcCurve(const std::vector<float> &tgcCurvePoints) override;
    void setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) override;
    void setTgcCurve(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic) override;
    std::vector<float> getTgcCurvePoints(float maxT) const override;
    void setPgaGain(uint16 value) override;
    uint16 getPgaGain() override;
    void setLnaGain(uint16 value) override;
    uint16 getLnaGain() override;
    void setLpfCutoff(uint32 value) override;
    void setDtgcAttenuation(std::optional<uint16> value) override;
    void setActiveTermination(std::optional<uint16> value) override;
    void start() override;
    void stop() override;
    float getSamplingFrequency() const override;
    float getCurrentSamplingFrequency() const override;
    void setHpfCornerFrequency(uint32_t frequency) override;
    void disableHpf() override;

private:
    using DataFrame = std::vector<int16_t>;

    std::vector<DataFrame> readDataset(const std::string &filepath);

    void producer();

    State state{State::STOPPED};
    Logger::Handle logger;
    std::mutex deviceStateMutex;
    std::thread producerThread;
    std::vector<DataFrame> dataset;
    size_t datasetSize{0};
    ProbeModel probeModel;
    arrus::framework::NdArray::Shape frameShape;
    std::optional<ops::us4r::Scheme> currentScheme;
    std::shared_ptr<DatasetBuffer> buffer;
};

}// namespace arrus::devices


#endif//ARRUS_CORE_DEVICES_DATASET_DATASETIMPL_H
