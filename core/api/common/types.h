#ifndef ARRUS_CORE_TYPES_H
#define ARRUS_CORE_TYPES_H

#include <vector>
#include <memory>

namespace arrus {
    // Data types
	using uint32 = unsigned int;
	using uint16 = unsigned short;
    using uint8 = unsigned char;

	using ChannelIdx = uint8;
	using BitMask = std::vector<bool>;
	using TGCSampleValue = float;
    using TGCCurve = std::vector<TGCSampleValue>;
}

#endif //ARRUS_CORE_TYPES_H
