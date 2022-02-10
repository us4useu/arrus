#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZER_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZER_H

#include "arrus/core/api/devices/DeviceId.h"
#include "IUs4OEMFactory.h"

namespace arrus::devices {

class IUs4OEMInitializer {
public:
    /**
     * Sorts the given list of us4oems (by device id).
     */
    virtual void sortModulesById(std::vector<IUs4OEMHandle> &ius4oems) = 0;

    /**
     * Initialize list of us4oem modules. Note: the list handles
     * should be already sorted by us4oem ID.
     */
    virtual void initModules(const std::vector<IUs4OEMHandle> &ius4oems) = 0;
    virtual ~IUs4OEMInitializer() = default;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZER_H
