#ifndef ARRUS_CORE_DEVICES_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4OEM_US4OEMIMPL_H

#include "arrus/core/devices/us4oem/Us4OEM.h"
#include "arrus/core/devices/us4oem/Us4OEM.h"
#include "arrus/core/external/ius4oem/IUs4OEMFactory.h"

namespace arrus {
class Us4OEMImpl : public Us4OEM {
public:
    Us4OEMImpl(const DeviceId id, IUs4OEMHandle &ius4eom)
    : Us4OEM(id), ius4oem(std::move(ius4eom)) {

    }

private:
    IUs4OEMHandle ius4oem;

};
}

#endif //ARRUS_CORE_DEVICES_US4OEM_US4OEMIMPL_H
