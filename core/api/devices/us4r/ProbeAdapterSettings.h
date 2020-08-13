#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/types.h"

namespace arrus {
class ProbeAdapterSettings {
public:
    using Us4OEMOrdinal = Ordinal;
    using ChannelAddress = std::pair<Us4OEMOrdinal, ChannelIdx>;
    using ChannelMapping = std::vector<ChannelAddress>;

    [[nodiscard]] const ChannelMapping &getChannelMapping() const {
        return mapping;
    }

private:
    ChannelMapping mapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
