#ifndef ARRUS_CORE_API_OPS_US4R_TGC_H
#define ARRUS_CORE_API_OPS_US4R_TGC_H

#include <vector>

namespace arrus::ops::us4r {

using TGCSampleValue = float;
/** TGC curve to apply on the us4r device. */
using TGCCurve = std::vector<TGCSampleValue>;

}

#endif //ARRUS_CORE_API_OPS_US4R_TGC_H
