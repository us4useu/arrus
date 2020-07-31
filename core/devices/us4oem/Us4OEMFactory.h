#ifndef ARRUS_CORE_DEVICES_US4OEM_US4OEMFACTORY_H
#define ARRUS_CORE_DEVICES_US4OEM_US4OEMFACTORY_H

#include "core/devices/DeviceId.h"
#include "core/devices/us4oem/Us4OEM.h"
#include "core/devices/us4oem/Us4OEMSettings.h"

#include "core/external/ius4oem/IUs4OEMFactory.h"

namespace arrus {

/**
 * This class should not be part of the c++ api! Due to dependency on IUs4OEMHandle
 *
 * To get an access to single Us4OEM: create a single-module Us4R custom system.
 */
class Us4OEMFactory {
public:
    virtual Us4OEM::Handle
    getUs4OEM(Ordinal ordinal, const IUs4OEMHandle &handle,
              const Us4OEMSettings &settings) = 0;
};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_US4OEMFACTORY_H
