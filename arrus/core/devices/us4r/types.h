#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_TYPES_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_TYPES_H

#include <unordered_map>

#include "arrus/core/api/common/types.h"
#include "arrus/core/common/hash.h"

namespace arrus::devices {
/** logical op id -> physical [start, end] ops */
using LogicalToPhysicalOp = std::vector<std::pair<OpId, OpId>>;
}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_TYPES_H
