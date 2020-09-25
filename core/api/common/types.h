#ifndef ARRUS_CORE_TYPES_H
#define ARRUS_CORE_TYPES_H

#include <vector>
#include <memory>

namespace arrus {
// Data types
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = int32_t;
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;

using float32 = float;
using float64 = double;

using ChannelIdx = uint16;
using BitMask = std::vector<bool>;
using Voltage = uint8;

template<typename T> using PtrHandle = T *;
}

#endif //ARRUS_CORE_TYPES_H
