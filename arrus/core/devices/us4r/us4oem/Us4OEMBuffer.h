#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/framework/NdArray.h"

namespace arrus::devices {

/**
 * A description of a single us4oem buffer element.
 *
 * An element is described by:
 * - src address
 * - size - size of the element (bytes)
 * - firing - a firing which ends the acquiring Tx/Rx sequence
 */
class Us4OEMBufferElement {
public:

    Us4OEMBufferElement(size_t address, size_t size, uint16 firing,
                        framework::NdArray::Shape elementShape,
                        framework::NdArray::DataType dataType)
                        : address(address), size(size), firing(firing),
                          elementShape(std::move(elementShape)), dataType(dataType) {}

    [[nodiscard]] size_t getAddress() const {
        return address;
    }

    [[nodiscard]] size_t getSize() const {
        return size;
    }

    [[nodiscard]] uint16 getFiring() const {
        return firing;
    }
    // TODO store ndarray here with placement on the us4oem module
    [[nodiscard]] const framework::NdArray::Shape &getElementShape() const {
        return elementShape;
    }

    [[nodiscard]] framework::NdArray::DataType getDataType() const {
        return dataType;
    }

private:
    size_t address;
    size_t size;
    uint16 firing;
    framework::NdArray::Shape elementShape;
    framework::NdArray::DataType dataType;
};

/**
 * A class describing a structure of a buffer that is located in the Us4OEM
 * memory.
 */
class Us4OEMBuffer {
public:
    explicit Us4OEMBuffer(std::vector<Us4OEMBufferElement> elements)
        : elements(std::move(elements)) {}

    [[nodiscard]] const Us4OEMBufferElement &getElement(size_t i) const {
        return elements[i];
    }

    [[nodiscard]] size_t getNumberOfElements() const {
        return elements.size();
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getElements() const {
        return elements;
    }

private:
    std::vector<Us4OEMBufferElement> elements;
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
