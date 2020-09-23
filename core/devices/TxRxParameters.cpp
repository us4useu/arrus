#include "TxRxParameters.h"

#include "arrus/core/api/common/types.h"
#include "arrus/core/common/collections.h"

namespace arrus::devices {

const TxRxParameters TxRxParameters::NOP = TxRxParameters(
    std::vector<bool>(),
    std::vector<float>(),
    ops::us4r::Pulse(0, 0, false),
    std::vector<bool>(),
    Interval<uint32>(0, 1),
    0, 0);

}