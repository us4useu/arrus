#ifndef ARRUS_CORE_DEVICES_FILE_FILEBUFFER_H
#define ARRUS_CORE_DEVICES_FILE_FILEBUFFER_H

#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/devices/file/FileBufferElement.h"
#include <iostream>

namespace arrus::devices {

class FileBuffer: public arrus::framework::DataBuffer {
public:

    FileBuffer(size_t nElements, const arrus::framework::NdArray::Shape& shape) {
        for(size_t i = 0; i < nElements; ++i) {
            elements.push_back(std::make_shared<FileBufferElement>(i, shape));
        }
    }

    ~FileBuffer() override = default;

    void close() {
        for(auto &e: elements) {
            e->close();
        }
    }

    bool write(size_t i, const std::function<void(framework::BufferElement::BufferElement::SharedHandle)> &func) {
        auto &element = elements.at(i);
        return element->write([&element, &func]() {
            func(element);
        });
    }

    bool read(size_t i, const std::function<void(framework::BufferElement::BufferElement::SharedHandle)> &func) {
        auto &element = elements.at(i);
        return element->read([&element, &func]() {
            func(element);
        });
    }

    void registerOnNewDataCallback(arrus::framework::OnNewDataCallback &callback) override {
        this->onNewDataCallback = callback;
    }


    std::shared_ptr<arrus::framework::BufferElement> getElement(size_t i) override {
        return elements.at(i);
    }

    size_t getElementSize() const override {
        if(elements.empty()) {
            throw std::runtime_error("The File Buffer is empty.");
        }
        return elements.at(0)->getSize();
    }

    size_t getNumberOfElementsInState(arrus::framework::BufferElement::State state) const override {
        int result = 0;
        for(auto &e: elements) {
            if(e->getState() == state) {
                ++result;
            }
        }
        return result;
    }

    void slice(size_t i, int begin, int end) {
        for(auto &e: elements) {
            e->slice(i, begin, end);
        }
    }

    size_t getNumberOfElements() const override { return elements.size(); }
    const framework::OnNewDataCallback &getOnNewDataCallback() const { return onNewDataCallback; }

    void registerOnOverflowCallback(arrus::framework::OnOverflowCallback&) override {/*Ignored*/}
    void registerShutdownCallback(arrus::framework::OnShutdownCallback&) override {/*Ignored*/}

private:
    std::vector<std::shared_ptr<FileBufferElement>> elements;
    // Blocking queue of elements to consume
    arrus::framework::OnNewDataCallback onNewDataCallback;
};

}
#endif//ARRUS_CORE_DEVICES_FILE_FILEBUFFER_H
