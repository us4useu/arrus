#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTER_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTER_H

#include <memory>
#include "arrus/core/api/devices/Device.h"

namespace arrus::devices {

class ProbeAdapter : public Device {
public:
    using Handle = std::unique_ptr<ProbeAdapter>;
    using RawHandle = PtrHandle<ProbeAdapter>;

    ~ProbeAdapter() override = default;

    ProbeAdapter(ProbeAdapter const&) = delete;
    ProbeAdapter(ProbeAdapter const&&) = delete;
    void operator=(ProbeAdapter const&) = delete;
    void operator=(ProbeAdapter const&&) = delete;

    [[nodiscard]] virtual ChannelIdx getNumberOfChannels() const = 0;
protected:
    explicit ProbeAdapter(const DeviceId &id): Device(id) {}
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTER_H
