#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEFACTORY_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEFACTORY_H

#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"

namespace arrus::devices {
class ProbeFactory {
public:
    virtual Probe::Handle
    getProbe(const ProbeSettings &settings,
             ProbeAdapter::RawHandle adapter) = 0;
};
}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEFACTORY_H
