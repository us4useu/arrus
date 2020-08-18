#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H

#include <arrus/core/api/devices/us4r/Us4OEM.h>
#include <arrus/core/api/devices/us4r/ProbeAdapterSettings.h>

#include <utility>
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"

namespace arrus {

class ProbeAdapterImpl : public ProbeAdapter {
public:
    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;
    using ChannelMapping = ProbeAdapterSettings::ChannelMapping;

    ProbeAdapterImpl(DeviceId deviceId, ProbeAdapterModelId modelId,
                     std::vector<Us4OEM::RawHandle> us4oems,
                     ChannelMapping channelMapping)
            : ProbeAdapter(deviceId), modelId(std::move(modelId)),
              us4oems(std::move(us4oems)),
              channelMapping(std::move(channelMapping)) {}

private:
    ProbeAdapterModelId modelId;
    std::vector<Us4OEM::RawHandle> us4oems;
    ChannelMapping channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
