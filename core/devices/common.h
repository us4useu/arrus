#ifndef ARRUS_CORE_DEVICES_TYPES_H
#define ARRUS_CORE_DEVICES_TYPES_H

#include <vector>
#include <ostream>

#include <boost/algorithm/string.hpp>

#include "Device.h"
#include "core/common/hash.h"

namespace arrus {

using DeviceHandle = std::unique_ptr<Device>;
using TGCCurve = std::vector<float>;

}
#endif //ARRUS_CORE_DEVICES_TYPES_H
