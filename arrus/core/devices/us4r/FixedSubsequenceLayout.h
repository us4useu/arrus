#ifndef ARRUS_CORE_DEVICES_US4R_FIXEDSUBSEQUENCELAYOUT_H
#define ARRUS_CORE_DEVICES_US4R_FIXEDSUBSEQUENCELAYOUT_H

#include <cstdlib>

namespace arrus::devices {

/**
 * Whether a sub-sequence keeps the data layout of the uploaded sequence ("fixed layout").
 *
 * By default the frames acquired by a sub-sequence are PACKED at the beginning of the output
 * buffer element, so the element size, the destination address of each frame and therefore the
 * DMA descriptors all depend on which TX/RXs are selected -- they have to be re-created on every
 * sub-sequence change (page-locking the host memory again costs several ms per us4OEM).
 *
 * With the fixed layout the element keeps the size of the uploaded sequence and each TX/RX is
 * transferred to ITS OWN slot, the one it would occupy in the full sequence. The transfers of a
 * TX/RX are then always the same, so their descriptor tables are reused (the driver caches them
 * by host address, size and us4OEM address), the output buffer never has to be re-allocated and
 * the processing pipeline never has to be re-created. The cost is the memory: an element is as
 * large as the full sequence's, no matter how few TX/RXs are selected.
 *
 * Experimental, enabled with ARRUS_FIXED_SUBSEQUENCE_LAYOUT=1.
 */
inline bool isFixedSubsequenceLayout() {
    static const bool value = []() {
        const char *env = std::getenv("ARRUS_FIXED_SUBSEQUENCE_LAYOUT");
        return env != nullptr && env[0] == '1';
    }();
    return value;
}

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_FIXEDSUBSEQUENCELAYOUT_H
