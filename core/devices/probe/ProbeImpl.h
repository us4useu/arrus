#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H

#include <arrus/core/api/devices/probe/ProbeModel.h>
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"

namespace arrus {

class ProbeImpl : public Probe {
public:
    ProbeImpl(const DeviceId &id, ProbeModel model,
              ProbeAdapter::RawHandle adapter,
              std::vector<ChannelIdx> channelMapping)
            : Probe(id), model(std::move(model)), adapter(adapter),
              channelMapping(std::move(channelMapping)) {}

private:
    ProbeModel model;
    ProbeAdapter::RawHandle adapter;
    std::vector<ChannelIdx> channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
