#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTER_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTER_H

#include <memory>
#include "arrus/core/api/devices/Device.h"

namespace arrus {

class ProbeAdapter : public Device {
public:
    using Handle = std::unique_ptr<ProbeAdapter>;

    ~ProbeAdapter() override = default;

    ProbeAdapter(ProbeAdapter const&) = delete;
    ProbeAdapter(ProbeAdapter const&&) = delete;
    void operator=(ProbeAdapter const&) = delete;
    void operator=(ProbeAdapter const&&) = delete;

protected:
    explicit ProbeAdapter(const DeviceId &id): Device(id) {}
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTER_H
