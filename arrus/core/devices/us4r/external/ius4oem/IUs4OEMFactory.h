#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORY_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORY_H

#include <memory>
#include <ius4oem.h>

#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::devices {

using IUs4OEMHandle = std::unique_ptr<IUs4OEM>;
using Ius4OEMRawHandle = IUs4OEM*;

/**
 * A simple wrapper over GetUs4OEM method available in Us4.
 */

class IUs4OEMFactory {
public:
    virtual IUs4OEMHandle getIUs4OEM(unsigned index) = 0;
    virtual std::vector<IUs4OEMHandle> getModules(Ordinal nModules) = 0;
    virtual ~IUs4OEMFactory() = default;
};


}
#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORY_H
