#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H

#include <ostream>
#include <unordered_set>
#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"
#include "arrus/core/api/devices/us4r/io/IOSettings.h"

namespace arrus::devices {
class ProbeAdapterSettings {
public:
    using Us4OEMOrdinal = Ordinal;
    using ChannelAddress = std::pair<Us4OEMOrdinal, ChannelIdx>;
    using ChannelMapping = std::vector<ChannelAddress>;

    ProbeAdapterSettings(ProbeAdapterModelId modelId, ChannelIdx nChannels, const ChannelMapping &mapping,
                         us4r::IOSettings ioSettings = us4r::IOSettings())
        : modelId(std::move(modelId)), nChannels(nChannels), mapping(mapping), ioSettings(std::move(ioSettings)) {}

    const ProbeAdapterModelId &getModelId() const { return modelId; }

    ChannelIdx getNumberOfChannels() const { return nChannels; }

    const ChannelMapping &getChannelMapping() const { return mapping; }

    const us4r::IOSettings &getIOSettings() const { return ioSettings; }

private:
    ProbeAdapterModelId modelId;
    ChannelIdx nChannels;
    ChannelMapping mapping;
    us4r::IOSettings ioSettings;
};
}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
