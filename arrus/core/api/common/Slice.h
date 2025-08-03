#ifndef ARRUS_CORE_API_COMMON_SLICE_H
#define ARRUS_CORE_API_COMMON_SLICE_H

#include <cstddef>

namespace arrus {

/**
 * Represents [start, end) slice.
 */
class Slice {
public:
    Slice(size_t start, size_t end, size_t step) : start(start), end(end), step(step) {}
    Slice(size_t start, size_t end): Slice(start, end, 1) {}

    size_t getStart() const { return start; }
    size_t getEnd() const { return end; }
    size_t getStep() const { return step; }

private:
    size_t start, end, step = 1;
};

}

#endif//RRUS_CORE_API_COMMON_SLICE_H
