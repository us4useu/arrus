#ifndef ARRUS_CORE_COMMON_APERTURE_H
#define ARRUS_CORE_COMMON_APERTURE_H

#include <vector>
#include <numeric>

#include "arrus/core/api/common/types.h"

namespace arrus {

inline
ChannelIdx getNumberOfActiveChannels(const BitMask &aperture) {
    return static_cast<ChannelIdx>(std::accumulate(std::begin(aperture), std::end(aperture),0));
}

}

#endif //ARRUS_CORE_COMMON_APERTURE_H
