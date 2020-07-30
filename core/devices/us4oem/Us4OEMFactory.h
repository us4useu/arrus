#ifndef ARRUS_CORE_DEVICES_US4OEM_US4OEMFACTORY_H
#define ARRUS_CORE_DEVICES_US4OEM_US4OEMFACTORY_H

#include "core/devices/DeviceId.h"
#include "core/devices/us4oem/Us4OEM.h"
#include "core/devices/us4oem/Us4OEMSettings.h"

namespace arrus {

class Us4OEMFactory {
    virtual Us4OEM::Handle
    getUs4OEM(Ordinal ordinal, const Us4OEMSettings &settings) = 0;
};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_US4OEMFACTORY_H
