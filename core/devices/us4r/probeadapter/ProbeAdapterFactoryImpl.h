#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H

#include "arrus/core/api/devices/probe/Probe.h"
#include "ProbeAdapterFactory.h"

namespace arrus {

class ProbeAdapterFactoryImpl : public ProbeAdapterFactory {
public:
    ProbeAdapter::Handle
    getProbeAdapter(const ProbeAdapterSettings &settings,
                    std::vector<Us4OEM::RawHandle> us4oems) override {
        // Validate
        // Make sure, that the number of us4oems is equal to
        // the number of us4oem mapping configurations available in the
        // provided mapping
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H
