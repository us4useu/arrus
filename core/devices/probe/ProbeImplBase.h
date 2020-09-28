#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEIMPLBASE_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEIMPLBASE_H

#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/devices/UltrasoundDevice.h"

namespace arrus::devices {

class ProbeImplBase : public Probe, public UltrasoundDevice {
    using Probe::Probe;
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEIMPLBASE_H
