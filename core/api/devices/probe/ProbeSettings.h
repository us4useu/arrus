#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H

#include <utility>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"

namespace arrus {
class ProbeSettings {
public:
    ProbeSettings(ProbeModel model,
                  std::vector<ChannelIdx> channelMapping)
            : model(std::move(model)),
              channelMapping(std::move(channelMapping)) {}

    [[nodiscard]] const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    [[nodiscard]] const ProbeModel &getModel() const {
        return model;
    }

private:
    ProbeModel model;
    /** A probe channel mapping to the underlying device. */
    std::vector<ChannelIdx> channelMapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
