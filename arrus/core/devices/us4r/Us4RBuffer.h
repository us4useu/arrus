#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H

#include <utility>
#include <vector>
#include <memory>
#include <iostream>

#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

class Us4RBufferElement {
public:
    explicit Us4RBufferElement(std::vector<Us4OEMBufferElement> elements): elements(std::move(elements)) {
        // Intentionally copying input shape.
        std::vector<size_t> shapeInternal = this->elements[0].getElementShape().getValues();
        // It's always the last axis, regardless IQ vs RF data.
        size_t channelAxis = shapeInternal.size()-1;

        auto nChannels = static_cast<unsigned>(shapeInternal[channelAxis]);
        unsigned nSamples = 0;
        framework::NdArray::DataType dataType = this->elements[0].getDataType();

        // Sum buffer us4oem component number of samples to determine buffer element shape.
        for(auto& component: this->elements) {
            auto &componentShape = component.getElementShape();
            // Verify if we have the same number of channels for each component
            if(nChannels != componentShape.get(channelAxis)) {
                throw IllegalArgumentException("Each us4OEM buffer element should have the same number of channels.");
            }
            if(dataType != component.getDataType()) {
                throw IllegalArgumentException("Each us4OEM buffer element component should have the same data type.");
            }
            nSamples += static_cast<unsigned>(componentShape.get(0));
        }
        shapeInternal[0] = nSamples;
        // Possibly another dimension: 2 (DDC I/Q)
        shapeInternal[channelAxis] = nChannels;
        elementShape = framework::NdArray::Shape{shapeInternal};
        elementDataType = dataType;
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

    /**
     * Abstraction of the whole Us4R buffer.
     *
     * @param elements us4R elements
     * @param parts all parts of all arrays of a single element, parts[i] are the parts of the us4OEM:i buffer
     */
    explicit Us4RBuffer(std::vector<Us4RBufferElement> elements, std::vector<Us4OEMBufferElementParts> parts)
    : elements(std::move(elements)), parts(std::move(parts)) {}

    [[nodiscard]] const Us4RBufferElement &getElement(const size_t i) const {
        return elements[i];
    }

    [[nodiscard]] size_t getNumberOfElements() const {
        return elements.size();
    }

    [[nodiscard]] bool empty() {
        return elements.empty();
    }

    size_t getElementSize() {
        size_t result = 0;
        for(auto &element: elements[0].getUs4OEMElements()) {
            result += element.getSize();
        }
        return result;
    }

    [[nodiscard]] Us4OEMBuffer getUs4oemBuffer(Ordinal ordinal) const {
        std::vector<Us4OEMBufferElement> us4oemBufferElements;
        for(const auto &element : elements) {
            us4oemBufferElements.push_back(element.getUs4OEMElement(ordinal));
        }
        return Us4OEMBuffer{us4oemBufferElements, parts[ordinal]};
    }

private:
    std::vector<Us4RBufferElement> elements;
    std::vector<Us4OEMBufferElementParts> parts;
};

class Us4RBufferBuilder {
public:
    void pushBack(const Us4OEMBuffer &us4oemBuffer) {
        if(!elements.empty() && elements.size() != us4oemBuffer.getNumberOfElements()) {
            throw ArrusException("Each Us4OEM rx buffer should have the same number of elements.");
        }
        if(elements.empty()) {
            elements = std::vector<std::vector<Us4OEMBufferElement>>{us4oemBuffer.getNumberOfElements()};
            parts = std::vector<std::vector<Us4OEMBufferElementPart>>{};
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
    std::vector<std::vector<Us4OEMBufferElementPart>> parts;
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
