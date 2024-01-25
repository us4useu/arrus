#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H

#include <utility>
#include <vector>
#include <memory>
#include <iostream>

#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

class Us4RBufferArrayDef {
public:
    [[nodiscard]] const framework::NdArrayDef &getDefinition() const { return definition; }
    [[nodiscard]] size_t getAddress() const { return address; }
    [[nodiscard]] const std::vector<size_t> &getOEMSizes() const { return oemSizes; }

private:
    framework::NdArrayDef definition;
    size_t address{0};
    std::vector<size_t> oemSizes;
};

class Us4RBufferElement {
public:
    explicit Us4RBufferElement(std::vector<Us4OEMBufferElement> elements): elements(std::move(elements)) {

    }

    [[nodiscard]] const Us4OEMBufferElement &getUs4OEMElement(const size_t ordinal) const {
        return elements[ordinal];
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getUs4OEMElements() const {
        return elements;
    }

    [[nodiscard]] size_t getNumberOfUs4OEMs() const {
        return elements.size();
    }

    [[nodiscard]] const framework::NdArray::Shape &getShape() const {
        return elementShape;
    }

    [[nodiscard]] framework::NdArray::DataType getDataType() const {
        return elementDataType;
    }

private:
    std::vector<Us4OEMBufferElement> elements;
    framework::NdArray::Shape elementShape{0, 0};
    framework::NdArray::DataType elementDataType;
};

class Us4RBuffer {
public:
    using Handle = std::unique_ptr<Us4RBuffer>;

    Us4RBuffer(unsigned nElements, const std::vector<Us4OEMBuffer> &buffers): nElements(nElements), buffers(buffers) {}

    [[nodiscard]] bool empty() const { return nElements == 0; }

    [[nodiscard]] size_t getNumberOfOEMs() const {return buffers.size(); }

    [[nodiscard]] Us4OEMBuffer getUs4OEMBuffer(Ordinal ordinal) const { return buffers.at(ordinal); }

private:
    unsigned nElements{0};
    std::vector<Us4OEMBuffer> buffers;
};

class Us4RBufferBuilder {
public:
    void pushBack(const Us4OEMBuffer &us4oemBuffer) {
        if(!elements.empty() && elements.size() != us4oemBuffer.getNumberOfElements()) {
            throw ArrusException("Each Us4OEM rx buffer should have the same number of elements.");
        }
        if(elements.empty()) {
            elements = std::vector<std::vector<Us4OEMBufferElement>>{us4oemBuffer.getNumberOfElements()};
            parts = std::vector<std::vector<Us4OEMBufferArrayPart>>{};
        }
        // Append us4oem buffer elements.
        for(size_t i = 0; i < us4oemBuffer.getNumberOfElements(); ++i) {
            elements[i].push_back(us4oemBuffer.getElement(i));
        }
        parts.push_back(us4oemBuffer.getElementParts());
    }

    Us4RBuffer::Handle build() {
        // Create buffer.
        std::vector<Us4RBufferElement> us4rElements;
        for(auto &element : elements) {
            us4rElements.emplace_back(element);
        }
        return std::make_unique<Us4RBuffer>(us4rElements, parts);
    }

private:
    // element number -> us4oem ordinal -> part of the buffer element
    std::vector<std::vector<Us4OEMBufferElement>> elements;
    // us4oem ordinal -> list of buffer element parts
    std::vector<std::vector<Us4OEMBufferArrayPart>> parts;
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
