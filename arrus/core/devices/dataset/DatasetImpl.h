#ifndef ARRUS_CORE_DEVICES_DATASET_DATASETIMPL_H
#define ARRUS_CORE_DEVICES_DATASET_DATASETIMPL_H

#include <array>
#include <arrus/core/api/framework/NdArray.h>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "arrus/core/api/devices/Ultrasound.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/session/Metadata.h"

namespace arrus::devices {

class DatasetBufferElement: public arrus::framework::BufferElement {
public:
    explicit DatasetBufferElement(std::vector<int16_t> data, size_t position, arrus::framework::NdArray::Shape shape)
        : data(std::move(data)),
          ndarray{data.data(), shape, arrus::framework::NdArray::DataType::INT16, DeviceId(DeviceType::CPU, 0)}
    {}

    ~DatasetBufferElement() override = default;
    void release() override {/*Ignored*/}
    framework::NdArray &getData() override { return ndarray; }
    size_t getSize() override { return data.size()*sizeof(int16_t); }
    size_t getPosition() override { return position; }
    State getState() const override { return State::READY; }
private:
    std::vector<int16_t> data;
    // NdArray - view for the above vector
    arrus::framework::NdArray ndarray;
    size_t position;
};

class DatasetBuffer: public arrus::framework::DataBuffer {
public:
    /**
     * @param data pointer to the place where the data starts
     * @param size the size of the buffer [number of bytes]
     */
    DatasetBuffer(std::vector<std::vector<int16_t>> frames, arrus::framework::NdArray::Shape shape){
        for(size_t i = 0; i < frames.size(); ++i) {
            elements.push_back(std::make_shared<DatasetBufferElement>(frames.at(i), i, shape));
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
    void producer();

    Logger::Handle logger;
    std::mutex deviceStateMutex;
    std::thread producerThread;
    std::string filepath;
    size_t datasetSize{0};
    ProbeModel probeModel;
    std::optional<ops::us4r::Scheme> currentScheme;
    std::shared_ptr<DatasetBuffer> buffer;
    State state{State::STOPPED};
};

}// namespace arrus::devices


#endif//ARRUS_CORE_DEVICES_DATASET_DATASETIMPL_H
