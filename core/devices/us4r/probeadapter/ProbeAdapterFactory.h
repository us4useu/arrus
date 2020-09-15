#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORY_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORY_H

#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "ProbeAdapterImpl.h"

namespace arrus::devices {
class ProbeAdapterFactory {
public:
    virtual ProbeAdapterImpl::Handle
    getProbeAdapter(const ProbeAdapterSettings &settings,
                    const std::vector<Us4OEMImpl::RawHandle> &us4oems) = 0;
};
}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORY_H
