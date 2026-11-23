#ifndef ARRUS_CORE_API_OPS_US4R_TGC_H
#define ARRUS_CORE_API_OPS_US4R_TGC_H

#include <vector>

#include "Us4OEM/api/Tgc.h"

namespace arrus::ops::us4r {

using TGCSampleValue = float;

using Tgc = ::us4us::us4r::Tgc;
using TgcBuilder = ::us4us::us4r::TgcBuilder;

/**
 * TGC curve to apply on the us4r device.
 *
 * @deprecated TGCCurve will be removed in ARRUS/ us4us HAL 2.0.0, please use arrus::ops::us4r::Tgc instead.
 */
using TGCCurve = Tgc;

}

#endif //ARRUS_CORE_API_OPS_US4R_TGC_H
