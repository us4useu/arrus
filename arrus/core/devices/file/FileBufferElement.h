#ifndef ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H
#define ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H

#include "arrus/core/api/framework/Buffer.h"

namespace arrus::devices {

class FileBufferElement: public arrus::framework::BufferElement {
public:
    FileBufferElement(size_t position, const arrus::framework::NdArray::Shape& shape) {
        this->size = shape.product(); // The number of int16 elements.
        this->data = new int16_t[size];
        this->ndarray = arrus::framework::NdArray{
            this->data,
            shape,
            arrus::framework::NdArray::DataType::INT16,
            DeviceId(DeviceType::CPU, 0)
        };
        this->position = position;
    }

    ~FileBufferElement() override = default;

    void acquire(const std::function<void(arrus::framework::BufferElement::BufferElement::SharedHandle)> &) {
        std::unique_lock<std::mutex> lock{stateMutex};
        // Wait until the element is free.
        readyForWrite.wait(lock, [this](){return this->state == arrus::framework::BufferElement::State::FREE;});
    }

    void releaseForRead() {
        std::unique_lock<std::mutex> lock{stateMutex};
    }

    void release() override {
        // Release
        readyForWrite.notify_one();
    }
    arrus::framework::NdArray &getData() override { return ndarray; }
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
    State state{arrus::framework::BufferElement::State::FREE};
};

}
#endif//ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H
