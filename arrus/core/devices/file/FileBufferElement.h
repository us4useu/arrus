#ifndef ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H
#define ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H

#include "arrus/core/api/framework/Buffer.h"

namespace arrus::devices {

class FileBufferElement: public arrus::framework::BufferElement {
public:

    FileBufferElement(size_t position, const arrus::framework::NdStorage::Shape& shape) {
        this->size = shape.product(); // The number of int16 elements.
        this->data = new int16_t[size];
        this->ndarray = arrus::framework::NdStorage{
            this->data,
            shape,
            arrus::framework::NdStorage::DataType::INT16,
            DeviceId(DeviceType::CPU, 0)
        };
        this->dataView = this->ndarray.view();
        this->position = position;
    }

    ~FileBufferElement() override {
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

    framework::NdStorage &getData() override { return dataView; }
    framework::NdStorage &getData(ArrayId) override {
        throw ArrusException("get data (ordinal) for file device is not implemented.");
    }
    uint16 getNumberOfArrays() const override {
        return 1;
    }

    arrus::framework::NdStorage &getAllData() {return ndarray; }
    size_t getSize() override { return size*sizeof(int16_t); }
    size_t getPosition() override { return position; }
    State getState() const override { return state; }

private:
    std::mutex stateMutex;
    std::condition_variable readyForWrite;
    std::condition_variable readyForRead;
    int16_t *data{nullptr};
    size_t size;
    // NdStorage: view of the above data pointer.
    arrus::framework::NdStorage ndarray{
        data,
        arrus::framework::NdStorage::Shape{},
        arrus::framework::NdStorage::DataType::INT16,
        DeviceId{DeviceType::CPU, 0}
    };
    arrus::framework::NdStorage dataView;
    size_t position;
    State state{arrus::framework::BufferElement::State::FREE};
    bool isClosed{false};
};

}
#endif//ARRUS_CORE_DEVICES_FILE_FILEBUFFERELEMENT_H
