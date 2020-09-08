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
                     ChannelIdx numberOfChannels,
                     ChannelMapping channelMapping)
            : ProbeAdapter(deviceId), modelId(std::move(modelId)),
              us4oems(std::move(us4oems)),
              numberOfChannels(numberOfChannels),
              channelMapping(std::move(channelMapping)) {}

    [[nodiscard]] ChannelIdx getNumberOfChannels() const override {
        return numberOfChannels;
    }

private:
    ProbeAdapterModelId modelId;
    std::vector<Us4OEM::RawHandle> us4oems;
    ChannelIdx numberOfChannels;
    ChannelMapping channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
