#ifndef ARRUS_CORE_DEVICES_DEVICEIDHASHER_H
#define ARRUS_CORE_DEVICES_DEVICEIDHASHER_H

#include <sstream>
#include <unordered_map>

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/common/hash.h"

namespace arrus::devices {

MAKE_HASHER(DeviceId, t.getDeviceType(), t.getOrdinal())

}

#endif //ARRUS_CORE_DEVICES_DEVICEIDHASHER_H
