#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H

#include <utility>
#include <unordered_set>
#include <ostream>

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"

namespace arrus {
class ProbeAdapterSettings {
public:
    using Us4OEMOrdinal = Ordinal;
    using ChannelAddress = std::pair<Us4OEMOrdinal, ChannelIdx>;
    using ChannelMapping = std::vector<ChannelAddress>;

    ProbeAdapterSettings(ProbeAdapterModelId modelId,
                         ChannelIdx numberOfChannels, ChannelMapping mapping)
            : modelId(std::move(modelId)), nChannels(numberOfChannels),
              mapping(std::move(mapping)) {}

    [[nodiscard]] const ProbeAdapterModelId &getModelId() const {
        return modelId;
    }

    [[nodiscard]] ChannelIdx getNumberOfChannels() const {
        return nChannels;
    }

    [[nodiscard]] const ChannelMapping &getChannelMapping() const {
        return mapping;
    }



private:
    ProbeAdapterModelId modelId;
    ChannelIdx nChannels;
    ChannelMapping mapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
