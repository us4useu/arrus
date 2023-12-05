#ifndef ARRUS_CORE_TYPES_H
#define ARRUS_CORE_TYPES_H

#include <vector>
#include <memory>

namespace arrus {
// Data types
using boolean = bool;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = unsigned long long int;
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = long long int;
using float32 = float;
using float64 = double;



using ChannelIdx = uint16;
typedef std::vector<bool> BitMask;
using Voltage = uint8;
using BitstreamId = uint16;

template<typename T> using PtrHandle = T *;

}


#endif //ARRUS_CORE_TYPES_H
