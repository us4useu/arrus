#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H

#include "arrus/common/compiler.h"
#include "arrus/common/format.h"

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/framework/NdArray.h"

namespace arrus::devices {

class Us4OEMBufferElementPart {
public:
    Us4OEMBufferElementPart(const size_t address, const size_t size, const uint16 firing, const unsigned nSamples)
        : address(address), size(size), firing(firing), nSamples(nSamples) {}

    size_t getAddress() const { return address; }
    size_t getSize() const { return size; }
    uint16 getFiring() const { return firing; }
    unsigned getNSamples() const { return nSamples; }

private:
    size_t address;
    size_t size;
    uint16 firing;
    unsigned nSamples;
};

class Us4OEMBuffer;

/**
 * A description of a single us4oem buffer element (which is now a batch of sequences).
 *
 * An element is described by:
 * - src address
 * - size - size of the element (in bytes)
 * - firing - a firing which ends the acquiring Tx/Rx sequence
 */
class Us4OEMBufferElement {
public:
    Us4OEMBufferElement(size_t address, size_t viewSize, uint16 firing, const framework::NdArray::Shape &viewShape,
                        framework::NdArray::DataType dataType)
        : address(address), viewSize(viewSize), firing(firing), viewShape(viewShape), dataType(dataType) {}

    /**
     * Returns the address OF THE WHOLE BUFFER ELEMENT (i.e. [0, nParts)). Byte-addressing.
     */
    [[nodiscard]] size_t getAddress() const { return address; }

    /**
     * Returns the size of THIS BUFFER ELEMENT View (i.e. [start, end]). Number of bytes.
     */
    [[nodiscard]] size_t getViewSize() const { return viewSize; }

    /**
     * Last firing of this element (when considering the WHOLE BUFFER ELEMENT).
     */
    [[nodiscard]] uint16 getFiring() const { return firing; }

    /**
     * Returns the shape of THIS BUFFER ELEMENT View (i.e. [start, end]).
     */
    [[nodiscard]] const framework::NdArray::Shape &getViewShape() const { return viewShape; }

    [[nodiscard]] framework::NdArray::DataType getDataType() const { return dataType; }

private:
    friend class Us4OEMBuffer;
    size_t address;
    size_t viewSize;
    uint16 firing;
    framework::NdArray::Shape viewShape;
    framework::NdArray::DataType dataType;
};

/**
 * A class describing a structure of a buffer that is located in the Us4OEM
 * memory.
 *
 * - All elements are assumed to have the same parts.
 * - All part addresses are relative to the beginning of each element.
 */
class Us4OEMBuffer {
public:
    static framework::NdArray::Shape getShape(bool isDDCOn, unsigned int totalNSamples, unsigned nChannels) {
        framework::NdArray::Shape result;
        if (isDDCOn) {
            result = {totalNSamples, 2, nChannels};
        } else {
            result = {totalNSamples, nChannels};
        }
        return result;
    }

    static framework::NdArray::Shape getShape(const framework::NdArray::Shape &currentShape, unsigned int totalNSamples) {
        if(currentShape.size() != 2 && currentShape.size() != 3) {
            throw std::runtime_error("Illegal us4OEM output buffer element shape order: " + std::to_string(currentShape.size()));
        }
        return getShape(currentShape.size() == 3, totalNSamples, currentShape.get(currentShape.size()-1));
    }

    explicit Us4OEMBuffer(std::vector<Us4OEMBufferElement> elements, std::vector<Us4OEMBufferElementPart> elementParts)
        : elements(std::move(elements)), elementParts(std::move(elementParts)) {}

    [[nodiscard]] const Us4OEMBufferElement &getElement(size_t i) const { return elements[i]; }

    [[nodiscard]] size_t getNumberOfElements() const { return elements.size(); }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getElements() const { return elements; }

    [[nodiscard]] const std::vector<Us4OEMBufferElementPart> &getElementParts() const { return elementParts; }

    /**
     * Returns the view of this buffer for slice [start, end] (note: end is inclusive).
     */
    Us4OEMBuffer getView(uint16 start, uint16 end) const {
        if (start > end) {
            throw IllegalArgumentException("Us4OEMBufferView: start cannot exceed end");
        }
        if (end >= elementParts.size()) {
            throw IllegalArgumentException(
                format("The index is outside of the scope of us4OEM Buffer view (size: {})", end, elementParts.size()));
        }
        auto b = std::begin(elementParts);

        // NOTE: +1 because end is inclusive
        std::vector<Us4OEMBufferElementPart> newParts(b + start, b + end + 1);
        std::vector<Us4OEMBufferElement> newElements;
        size_t oldSize = getUniqueElementSize(elements);
        IGNORE_UNUSED(oldSize);
        auto oldShape = getUniqueShape(elements);
        size_t newSize = std::accumulate(std::begin(newParts), std::end(newParts), 0,
            [](const auto &a, const auto &b){return a + b.getSize();});
        unsigned newNSamples = std::accumulate(std::begin(newParts), std::end(newParts), 0,
            [](const auto &a, const auto &b){return a + b.getNSamples();});
        auto newShape = getShape(oldShape, newNSamples);
        for(const auto &oldElement: elements) {
            Us4OEMBufferElement newElement(oldElement);
            newElement.viewSize = newSize;
            newElement.viewShape = newShape;
            newElements.push_back(newElement);
        }
        return Us4OEMBuffer(newElements, newParts);
    }

private:
    size_t getUniqueElementSize(const std::vector<Us4OEMBufferElement> &elements) const {
        std::unordered_set<size_t> sizes;
        for (auto &element: elements) {
            sizes.insert(element.getViewSize());
        }
        if (sizes.size() > 1) {
            throw ArrusException("Each us4oem buffer element should have the same size.");
        }
        // This is the size of a single element produced by this us4oem.
        const size_t elementSize = *std::begin(sizes);
        return elementSize;
    }

    framework::NdArray::Shape getUniqueShape(const std::vector<Us4OEMBufferElement> &elements) const {
        if(elements.empty()) {
            throw std::runtime_error("List of elements cannot be empty");
        }
        auto &shape = elements.at(0).getViewShape();
        for(size_t i = 0; i < elements.size(); ++i) {
            auto &otherShape = elements.at(i).getViewShape();
            if(shape != otherShape) {
                throw IllegalArgumentException("The shape of element's NdArray must be unique for each us4OEM.");
            }
        }
        return shape;
    }

    std::vector<Us4OEMBufferElement> elements;
    std::vector<Us4OEMBufferElementPart> elementParts;
};

}// namespace arrus::devices

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
