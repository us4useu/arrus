#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H

#include <utility>
#include <vector>
#include <memory>

#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

class Us4RBufferElement {
public:
    explicit Us4RBufferElement(std::vector<Us4OEMBufferElement> us4OemElements)
        : us4oemElements(std::move(us4OemElements)) {}

    [[nodiscard]] const Us4OEMBufferElement &getUs4oemElement(const size_t ordinal) const {
        return us4oemElements[ordinal];
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getUs4oemElements() const {
        return us4oemElements;
    }

    [[nodiscard]] size_t getNumberOfUs4oems() const {
        return us4oemElements.size();
    }

private:
    std::vector<Us4OEMBufferElement> us4oemElements;
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
        for(auto &element: elements[0].getUs4oemElements()) {
            result += element.getSize();
        }
        return result;
    }

    [[nodiscard]] Us4OEMBuffer getUs4oemBuffer(Ordinal ordinal) const {
        std::vector<Us4OEMBufferElement> us4oemBufferElements;
        for(const auto &element : elements) {
            us4oemBufferElements.push_back(element.getUs4oemElement(ordinal));
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
