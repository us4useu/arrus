#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBE_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBE_H

#include "../Device.h"

namespace arrus::devices {

class Probe : public Device {
public:
    using Handle = std::unique_ptr<Probe>;

    virtual ~Probe() = default;

    Probe(Probe const&) = delete;
    Probe(Probe const&&) = delete;
    void operator=(Probe const&) = delete;
    void operator=(Probe const&&) = delete;
protected:
    explicit Probe(const DeviceId &id): Device(id) {}

};

}



#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBE_H