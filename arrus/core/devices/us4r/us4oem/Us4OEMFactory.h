#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORY_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORY_H

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImplBase.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"

#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"

namespace arrus::devices {

/**
 * This class should not be part of the c++ api! Due to dependency on IUs4OEMHandle
 *
 * To get an access to single Us4OEM: create a single-module Us4R custom system.
 */
class Us4OEMFactory {
public:
    virtual Us4OEMImplBase::Handle
    getUs4OEM(Ordinal ordinal, IUs4OEMHandle &handle,
              const Us4OEMSettings &settings) = 0;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORY_H
