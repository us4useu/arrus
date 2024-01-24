#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H

#include "arrus/common/utils.h"

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/framework/NdArray.h"
#include "arrus/core/common/collections.h"

namespace arrus::devices {

/**
 * A single Us4OEM element part currently corresponds to a single entry sequencer
 * output, that is, a single RF frame (number of samples, number of us4OEM RX channels e.g. 32).
 * Note: the size can be 0 -- this kind of Part is to inform
 * that no transfer is performed in the given sequencer entry.
 */
class Us4OEMBufferElementPart {
public:
    Us4OEMBufferElementPart(size_t address, size_t size, uint16 arrayId, uint16 entryId)
        : address(address), size(size), arrayId(arrayId), entryId(entryId) {}

    size_t getAddress() const { return address;}
    size_t getSize() const { return size; }

    uint16 getArrayId() const { return arrayId; }

    uint16 getEntryId() const {return entryId;}

private:
    size_t address;
    size_t size;
    uint16 arrayId;
    uint16 entryId;
};
// A single array consits of multiple parts (frames).
using Us4OEMBufferArrayParts = std::vector<Us4OEMBufferElementPart>;
// A single element consists of 0 or more arrays.
using Us4OEMBufferElementParts = std::vector<Us4OEMBufferArrayParts>;

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

    Us4OEMBufferElement(size_t address, size_t size, uint16 firing, Tuple<framework::NdArrayDef> arrays)
                        : address(address), size(size), firing(firing), arrays(std::move(arrays)) {}

    [[nodiscard]] size_t getAddress() const {
        return address;
    }

    [[nodiscard]] size_t getSize() const {
        return size;
    }

    [[nodiscard]] uint16 getFiring() const {
        return firing;
    }

    const framework::NdArrayDef &getArray(ArrayId id) {
        return arrays.get(id);
    }

private:
    size_t address;
    size_t size;
    uint16 firing;
    Tuple<framework::NdArrayDef> arrays;
};

class Us4OEMBufferBuilder;

/**
 * A class describing a structure of a buffer that is located in the Us4OEM
 * memory.
 * - All elements are assumed to have the same number of arrays.
 * - All elements are assumed to have the same parts.
 * - All part addresses are relative to the beginning of each element.
 */
class Us4OEMBuffer {
public:
    explicit Us4OEMBuffer(std::vector<Us4OEMBufferElement> elements,
                          std::vector<std::vector<Us4OEMBufferElementPart>> elementParts)
        : elements(std::move(elements)), arrayParts(std::move(elementParts)) {}

    [[nodiscard]] const Us4OEMBufferElement &getElement(size_t i) const {
        return elements[i];
    }

    [[nodiscard]] size_t getNumberOfElements() const {
        return elements.size();
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getElements() const {
        return elements;
    }

    [[nodiscard]] ArrayId getNumberOfArrays() const {
        return ARRUS_SAFE_CAST(arrayParts.size(), ArrayId);
    }

    [[nodiscard]] const std::vector<Us4OEMBufferElementPart> &getParts(ArrayId arrayId) const {
        ARRUS_REQUIRES_TRUE_IAE(arrayId < arrayParts.size(), "Array number out of the bounds.");
        return arrayParts.at(arrayId);
    }

private:
    std::vector<Us4OEMBufferElement> elements;
    /** array Id -> list of array parts (frames) */
    std::vector<std::vector<Us4OEMBufferElementPart>> arrayParts;
};

class Us4OEMBufferBuilder {
public:
    void add(Us4OEMBufferElement element) {
        elements.emplace_back(std::move(element));
    }

    void add(Us4OEMBufferElementPart part) {
        if(mapContains(part.getArrayId())) {

        }
    }

private:
    std::vector<Us4OEMBufferElement> elements;
    std::vector<std::vector<Us4OEMBufferElementPart>> parts;
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
