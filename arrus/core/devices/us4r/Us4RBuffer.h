#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H

#include <utility>
#include <vector>

#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

class Us4RElement {
public:
    explicit Us4RElement(std::vector<Us4OEMBufferElement> us4OemElements)
        : us4oemElements(std::move(us4OemElements)) {}

    [[nodiscard]] const Us4OEMBufferElement &getUs4oemElement(const size_t i) const {
        return us4oemElements[i];
    }

    size_t getNumberOfUs4oems() {
        return us4oemElements.size();
    }

private:
    std::vector<Us4OEMBufferElement> us4oemElements;
};

class Us4RBuffer {
public:
    explicit Us4RBuffer(std::vector<Us4RElement> elements)
    : elements(std::move(elements)) {}

    [[nodiscard]] const Us4RElement &getElement(const size_t i) const {
        return elements[i];
    }

    [[nodiscard]] size_t getNumberOfElements() const {
        return elements.size();
    }

private:
    std::vector<Us4RElement> elements;
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4RBUFFER_H
