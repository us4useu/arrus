#include "TxRxParameters.h"

#include "arrus/core/api/common/types.h"
#include "arrus/core/common/collections.h"

namespace arrus::devices::us4r {

const TxRxParameters TxRxParameters::US4OEM_NOP = TxRxParameters(
    std::vector<bool>(128, false),
    std::vector<float>(128, 0),
    ops::us4r::Pulse(1e6, 1, false),
    std::vector<bool>(128, false),
    Interval<uint32>(0, 64),
    0, 80);






}