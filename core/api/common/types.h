#ifndef ARRUS_CORE_TYPES_H
#define ARRUS_CORE_TYPES_H

#include <vector>
#include <memory>

namespace arrus {
    // Data types
	using uint32 = unsigned int;
	using uint16 = unsigned short;
    using uint8 = unsigned char;

	using ChannelIdx = uint16;
	using BitMask = std::vector<bool>;
    using TGCCurve = std::vector<float>;
}

#endif //ARRUS_CORE_TYPES_H
