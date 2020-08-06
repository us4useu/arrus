#ifndef ARRUS_CORE_DEVICES_ADAPTER_ADAPTER_H
#define ARRUS_CORE_DEVICES_ADAPTER_ADAPTER_H

#include "arrus/core/devices/Device.h"

namespace arrus {

class Adapter : Device {
public:

    using Handle = std::unique_ptr<Adapter>;

    explicit Adapter(const DeviceId &id): Device(id) {}

    virtual ~Adapter() = default;

    Adapter(Adapter const&) = delete;
    Adapter(Adapter const&&) = delete;
    void operator=(Adapter const&) = delete;
    void operator=(Adapter const&&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_ADAPTER_ADAPTER_H
