#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H

#include <utility>
#include <vector>
#include <memory>

#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

class Us4RBufferElement {
public:
    explicit Us4RBufferElement(
        std::vector<Us4OEMBufferElement> us4oemComponents)
        : us4oemComponents(std::move(us4oemComponents)) {

        // Sum buffer us4oem component number of samples to determine buffer element shape.
        unsigned nChannels = us4oemComponents[0].getElementShape().get(1);
        unsigned nSamples = 0;
        framework::NdArray::DataType dataType = us4oemComponents[0].getDataType();

        for(auto& component: us4oemComponents) {
            auto &componentShape = component.getElementShape();
            // Verify if we have the same number of channels for each component
            if(nChannels != componentShape.get(1)) {
                throw IllegalArgumentException(
                    "Each Us4R buffer element component should have the same number of channels.");
            }
            if(dataType != component.getDataType()) {
                throw IllegalArgumentException(
                    "Each Us4R buffer element component should have the same data type.");
            }
            nSamples += componentShape.get(0);
        }
        elementShape = framework::NdArray::Shape{nChannels, nSamples};
        elementDataType = dataType;
    }

    [[nodiscard]] const Us4OEMBufferElement &getUs4oemComponent(const size_t ordinal) const {
        return us4oemComponents[ordinal];
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getUs4oemComponents() const {
        return us4oemComponents;
    }

    [[nodiscard]] size_t getNumberOfUs4oems() const {
        return us4oemComponents.size();
    }

    [[nodiscard]] const framework::NdArray::Shape &getShape() const {
        return elementShape;
    }

    [[nodiscard]] framework::NdArray::DataType getDataType() const {
        return elementDataType;
    }

private:
    std::vector<Us4OEMBufferElement> us4oemComponents;
    framework::NdArray::Shape elementShape{0, 0};
    framework::NdArray::DataType elementDataType;
};

class Us4RBuffer {
public:
    using Handle = std::unique_ptr<Us4RBuffer>;

    explicit Us4RBuffer(std::vector<Us4RBufferElement> elements)
    : elements(std::move(elements)) {}

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
        for(auto &element: elements[0].getUs4oemComponents()) {
            result += element.getSize();
        }
        return result;
    }

    [[nodiscard]] Us4OEMBuffer getUs4oemBuffer(Ordinal ordinal) const {
        std::vector<Us4OEMBufferElement> us4oemBufferElements;
        for(const auto &element : elements) {
            us4oemBufferElements.push_back(element.getUs4oemComponent(ordinal));
        }
        return Us4OEMBuffer(us4oemBufferElements);
    }

private:
    std::vector<Us4RBufferElement> elements;
};

class Us4RBufferBuilder {
public:
    void pushBackUs4oemBuffer(const Us4OEMBuffer &us4oemBuffer) {
        if(!elements.empty() && elements.size() != us4oemBuffer.getNumberOfElements()) {
            throw arrus::ArrusException("Each Us4OEM rx buffer should have the same number of elements.");
        }
        if(elements.empty()) {
            elements = std::vector<std::vector<Us4OEMBufferElement>>(us4oemBuffer.getNumberOfElements());
        }
        // Append us4oem buffer elements.
        for(size_t i = 0; i < us4oemBuffer.getNumberOfElements(); ++i) {
            elements[i].push_back(us4oemBuffer.getElement(i));
        }
    }

    Us4RBuffer::Handle build() {
        // Create buffer.
        std::vector<Us4RBufferElement> us4rElements;
        for(auto &element : elements) {
            us4rElements.emplace_back(element);
        }
        return std::make_unique<Us4RBuffer>(us4rElements);
    }

private:
    // element number -> us4oem ordinal -> part of the buffer element
    std::vector<std::vector<Us4OEMBufferElement>> elements;

};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
