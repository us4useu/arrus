#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "core/devices/Device.h"
#include "core/devices/us4oem/Us4OEM.h"

namespace arrus {

class Us4R : public Device {
public:
    using Handle = std::unique_ptr<Us4R>;

    explicit Us4R(const DeviceId &id): Device(id) {}

    virtual ~Us4R() = default;

    virtual Us4OEM::Handle &getUs4OEM(Ordinal ordinal) = 0;

    Us4R(Us4R const&) = delete;
    Us4R(Us4R const&&) = delete;
    void operator=(Us4R const&) = delete;
    void operator=(Us4R const&&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4R_H
