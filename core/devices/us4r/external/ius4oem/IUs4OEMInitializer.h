#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZER_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZER_H

#include "arrus/core/api/devices/DeviceId.h"
#include "IUs4OEMFactory.h"

namespace arrus {

class IUs4OEMInitializer {
public:
    /**
     * Sorts the given list of us4oems (by device id) and initializes them.
     */
    virtual void
    initModules(std::vector<IUs4OEMHandle> &ius4oems) = 0;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZER_H
