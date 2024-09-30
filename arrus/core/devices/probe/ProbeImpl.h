#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H

#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"

namespace arrus::devices {

class ProbeImpl : public Probe {
public:
    using Handle = std::unique_ptr<ProbeImpl>;
    using RawHandle = PtrHandle<ProbeImpl>;

    ProbeImpl(const DeviceId &id, ProbeModel model): Probe(id), model(std::move(model)) {}

    const ProbeModel &getModel() const override {
        return model;
    }

private:
    ProbeModel model;
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
