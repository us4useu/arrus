#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/framework/NdArray.h"

namespace arrus::devices {

/**
 * A single Us4OEM element part currently corresponds to a single entry sequencer
 * output, that is, a single RF frame (number of samples, number of us4OEM RX channels e.g. 32).
 * Note: the size can be 0 -- this kind of Part is to inform
 * that no transfer is performed in the given sequencer entry.
 */
class Us4OEMBufferElementPart {
public:
    Us4OEMBufferElementPart(size_t address, size_t size, uint16 entryId)
            : address(address), size(size), entryId(entryId) {}

    size_t getAddress() const {
        return address;
    }
    size_t getSize() const {
        return size;
    }
    uint16 getEntryId() const {
        return entryId;
    }

private:
    size_t address;
    size_t size;
    uint16 entryId;
};

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

    Us4OEMBufferElement(size_t address, size_t size, uint16 firing, framework::NdArray::Shape elementShape,
                        framework::NdArray::DataType dataType)
                        : address(address), size(size), firing(firing), elementShape(std::move(elementShape)),
                        dataType(dataType) {}

    [[nodiscard]] size_t getAddress() const {
        return address;
    }

    [[nodiscard]] size_t getSize() const {
        return size;
    }

    [[nodiscard]] uint16 getFiring() const {
        return firing;
    }

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

class Us4OEMBufferBuilder;

/**
 * A class describing a structure of a buffer that is located in the Us4OEM
 * memory.
 *
 * - All elements are assumed to have the same parts.
 * - All part addresses are relative to the beginning of each element.
 */
class Us4OEMBuffer {
public:
    explicit Us4OEMBuffer(std::vector<Us4OEMBufferElement> elements,
                          std::vector<Us4OEMBufferElementPart> elementParts)
        : elements(std::move(elements)), elementParts(std::move(elementParts)) {}

    [[nodiscard]] const Us4OEMBufferElement &getElement(size_t i) const {
        return elements[i];
    }

    [[nodiscard]] size_t getNumberOfElements() const {
        return elements.size();
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getElements() const {
        return elements;
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElementPart> &getElementParts() const {
        return elementParts;
    }

private:
    std::vector<Us4OEMBufferElement> elements;
    std::unordered_map<SequenceId, std::vector<Us4OEMBufferElementPart>> elementParts;
};

class Us4OEMBufferBuilder {
public:
    void add() {

    }
private:
    std::vector<Us4OEMBufferElement> elements;
    std::unordered_map<SequenceId, std::vector<Us4OEMBufferElementPart>> elementParts;
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
