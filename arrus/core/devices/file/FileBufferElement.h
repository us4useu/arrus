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
        this->dataView = this->ndarray.view();
        this->position = position;
    }

    ~FileBufferElement() {
        delete data;
    }

    bool write(const std::function<void()> &func) {
        std::unique_lock<std::mutex> lock{stateMutex};

        while(this->state == framework::BufferElement::State::READY) {
            readyForWrite.wait(lock);
        }
        if(this->state != framework::BufferElement::State::FREE) {
            return false;
        }
        func();
        this->state = framework::BufferElement::State::READY;
        readyForRead.notify_one();
        return true;
    }

    bool read(const std::function<void()> &func) {
        std::unique_lock<std::mutex> lock{stateMutex};
        while(this->state == framework::BufferElement::State::FREE) {
            readyForRead.wait(lock);
        }
        if(this->state != framework::BufferElement::State::READY) {
            return false;
        }
        lock.unlock();
        func();
        return true;
    }

    void release() override {
        std::unique_lock<std::mutex> lock{stateMutex};
        this->state = framework::BufferElement::State::FREE;
        readyForWrite.notify_one();
    }

    void close() {
        std::unique_lock<std::mutex> lock{stateMutex};
        this->state = framework::BufferElement::State::INVALID;
        readyForWrite.notify_all();
        readyForRead.notify_all();
    }

    void slice(size_t i, int begin, int end) {
        this->dataView = ndarray.slice(i, begin, end);
    }

    arrus::framework::NdArray &getData() override { return dataView; }
    size_t getSize() override { return size*sizeof(int16_t); }
    size_t getPosition() override { return position; }
    State getState() const override { return state; }
private:
    std::mutex stateMutex;
    std::condition_variable readyForWrite;
    std::condition_variable readyForRead;
    int16_t *data{nullptr};
    size_t size;
    // NdArray: view of the above data pointer.
    arrus::framework::NdArray ndarray{
        data,
        arrus::framework::NdArray::Shape{},
        arrus::framework::NdArray::DataType::INT16,
        DeviceId{DeviceType::CPU, 0}
    };
    arrus::framework::NdArray dataView;
    size_t position;
    State state{arrus::framework::BufferElement::State::FREE};
    bool isClosed{false};
};

}
#endif//ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H
