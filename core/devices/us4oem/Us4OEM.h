#ifndef ARRUS_CORE_DEVICES_US4OEM_H
#define ARRUS_CORE_DEVICES_US4OEM_H

#include <memory>
#include "core/devices/Device.h"
#include "core/common/types.h"

namespace arrus {

class Us4OEM : public Device {
public:
    using Handle = std::unique_ptr<Us4OEM>;

    Us4OEM(const DeviceId &id): Device(id) {}

    Us4OEM(Us4OEM const&) = delete;
    Us4OEM(Us4OEM const&&) = delete;
    void operator=(Us4OEM const&) = delete;
    void operator=(Us4OEM const&&) = delete;
};


}

#endif
