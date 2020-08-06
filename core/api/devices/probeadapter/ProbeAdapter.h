#ifndef ARRUS_CORE_DEVICES_ADAPTER_ADAPTER_H
#define ARRUS_CORE_DEVICES_ADAPTER_ADAPTER_H

#include "arrus/core/devices/Device.h"

namespace arrus {

class ProbeAdapter : Device {
public:
    using Handle = std::unique_ptr<ProbeAdapter>;

    explicit ProbeAdapter(const DeviceId &id): Device(id) {}

    virtual ~ProbeAdapter() = default;

    ProbeAdapter(ProbeAdapter const&) = delete;
    ProbeAdapter(ProbeAdapter const&&) = delete;
    void operator=(ProbeAdapter const&) = delete;
    void operator=(ProbeAdapter const&&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_ADAPTER_ADAPTER_H
