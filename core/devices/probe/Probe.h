#ifndef ARRUS_CORE_DEVICES_PROBE_PROBE_H
#define ARRUS_CORE_DEVICES_PROBE_PROBE_H

#include "core/devices/Device.h"

namespace arrus {

class Probe : Device {
    using Handle = std::unique_ptr<Probe>;

    explicit Probe(const DeviceId &id): Device(id) {}

    virtual ~Probe() = default;

    Probe(Probe const&) = delete;
    Probe(Probe const&&) = delete;
    void operator=(Probe const&) = delete;
    void operator=(Probe const&&) = delete;
};

}



#endif //ARRUS_CORE_DEVICES_PROBE_PROBE_H
