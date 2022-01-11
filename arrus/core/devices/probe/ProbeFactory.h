#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEFACTORY_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEFACTORY_H

#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/devices/probe/ProbeImplBase.h"

namespace arrus::devices {
class ProbeFactory {
public:
    virtual ProbeImplBase::Handle getProbe(const ProbeSettings &settings,ProbeAdapterImplBase::RawHandle adapter) = 0;
    virtual ~ProbeFactory() = default;
};
}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEFACTORY_H
