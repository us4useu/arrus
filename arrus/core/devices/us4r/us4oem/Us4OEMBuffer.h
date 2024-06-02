#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H

#include "arrus/common/compiler.h"
#include "arrus/common/format.h"
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
class Us4OEMBufferArrayPart {
public:
    Us4OEMBufferArrayPart(size_t address, size_t size, uint16 arrayId, uint16 entryId)
        : address(address), size(size), arrayId(arrayId), entryId(entryId) {}

    /** Returns address of this part, relative to the beginning of the array */
    size_t getAddress() const { return address; }

    size_t getSize() const { return size; }

    uint16 getArrayId() const { return arrayId; }

    /** Returns GLOBAL firing id (i.e. relative to the beginning of all seqeuncer entries) */
    uint16 getEntryId() const { return entryId; }


private:
    size_t address;
    size_t size;
    uint16 arrayId;
    uint16 entryId;
};

// A single array consits of multiple parts (frames).
using Us4OEMBufferArrayParts = std::vector<Us4OEMBufferArrayPart>;
// A single element consists of 0 or more arrays.
using Us4OEMBufferElementParts = std::vector<Us4OEMBufferArrayParts>;

class Us4OEMBufferArrayDef {
public:
    Us4OEMBufferArrayDef(size_t address, framework::NdArrayDef definition, Us4OEMBufferArrayParts parts)
        : address(address), definition(std::move(definition)), parts(std::move(parts)) {}

    size_t getAddress() const { return address; }
    const framework::NdArrayDef &getDefinition() const { return definition; }
    const Us4OEMBufferArrayParts &getParts() const { return parts; }
    /** The number of bytes this OEM produces. */
    [[nodiscard]] size_t getSize() const {
        size_t result = 0;
        for (const auto &part : parts) {
            result += part.getSize();
        }
        return result;
    }

private:
    /** Array address, relative to the buffer element address. */
    size_t address;
    framework::NdArrayDef definition;
    Us4OEMBufferArrayParts parts;
};

/**
 * A description of a single us4oem buffer element (which is a tuple of batches of sequences).
 *
 * An element is described by:
 * - src address
 * - size - size of the element (in bytes)
 * - firing - a firing which ends the acquiring Tx/Rx sequence
 */
class Us4OEMBufferElement {
public:
    Us4OEMBufferElement(size_t address, size_t size, uint16 firing) : address(address), size(size), firing(firing) {}

    [[nodiscard]] size_t getAddress() const { return address; }

    [[nodiscard]] size_t getSize() const { return size; }

    [[nodiscard]] uint16 getFiring() const { return firing; }

private:
    size_t address;
    size_t size;
    uint16 firing;
};

class Us4OEMBufferBuilder;

/**
 * A class describing a structure of a buffer that is located in the Us4OEM
 * memory.
 * - All elements are assumed to have the same number of arrays.
 * - All elements are assumed to have the same parts.
 * - All part addresses are relative to the beginning of each element.
 *
 * NOTE: all OEM buffers should have the same number of array definitions, regardless
 * of whether this OEM acquires some data or not. For empty arrays, the array def should
 * indicate empty array.
 */
class Us4OEMBuffer {
public:
    explicit Us4OEMBuffer(std::vector<Us4OEMBufferElement> elements, std::vector<Us4OEMBufferArrayDef> arrayDefs)
        : elements(std::move(elements)), arrayDefs(std::move(arrayDefs)) {}

    [[nodiscard]] const Us4OEMBufferElement &getElement(size_t i) const { return elements[i]; }

    [[nodiscard]] size_t getNumberOfElements() const { return elements.size(); }

    [[nodiscard]] const std::vector<Us4OEMBufferElement> &getElements() const { return elements; }

    [[nodiscard]] ArrayId getNumberOfArrays() const { return ARRUS_SAFE_CAST(arrayDefs.size(), ArrayId); }

    [[nodiscard]] const std::vector<Us4OEMBufferArrayDef> &getArrayDefs() const { return arrayDefs; }

    [[nodiscard]] const Us4OEMBufferArrayParts &getParts(ArrayId arrayId) const {
        ARRUS_REQUIRES_TRUE_IAE(arrayId < arrayDefs.size(), "Array number out of the bounds.");
        return arrayDefs.at(arrayId).getParts();
    }

    /** array id -> array address, relative to the beginning of an element */
    [[nodiscard]] size_t getArrayAddressRelative(uint16 arrayId) const { return arrayDefs.at(arrayId).getAddress(); }

    /**
     * Returns the view of this buffer for slice [start, end] (note: end is inclusive).
     */
    Us4OEMBuffer getView(SequenceId sequenceId, uint16 start, uint16 end) const {
        throw std::runtime_error("NYI");
    }

private:
    std::vector<Us4OEMBufferElement> elements;
    /** Array id -> array defintion (NdArray defintion + parts) */
    std::vector<Us4OEMBufferArrayDef> arrayDefs;
};

class Us4OEMBufferBuilder {
public:

    void add(Us4OEMBufferArrayDef def) {
        arrays.emplace_back(def);
    }

    void add(Us4OEMBufferElement element) {
        elements.emplace_back(element);
    }

    Us4OEMBuffer build() {
        return Us4OEMBuffer{elements, arrays};
    }

private:
    std::vector<Us4OEMBufferElement> elements;
    // Array -> parts
    std::vector<Us4OEMBufferArrayDef> arrays;
};

}// namespace arrus::devices

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMBUFFER_H
