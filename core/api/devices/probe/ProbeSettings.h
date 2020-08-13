#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H


#include "arrus/core/api/common/types.h"

namespace arrus {
class ProbeSettings {
public:
    [[nodiscard]] const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

private:
    /** Probe channel -> Adapter channel mapping. */
    std::vector<ChannelIdx> channelMapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
