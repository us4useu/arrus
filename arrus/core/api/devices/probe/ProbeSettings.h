#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H

#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "std4us/Optional.hpp"
#include "std4us/StdInterop.hpp"

namespace arrus::devices {

class ProbeSettings {
public:
    using ChannelMapping = std::vector<ChannelIdx>;

    /**
     *
     * @param model
     * @param channelMapping flattened channel mappings. For 2-D array channel
     *    mapping is row major order.
     */
    ProbeSettings(ProbeModel model, std::vector<ChannelIdx> channelMapping)
            : model(std::move(model)),
              channelMapping(std::move(channelMapping)) {}

    ProbeSettings(ProbeModel model, std::vector<ChannelIdx> channelMapping,
                  const std::optional<BitstreamId> &bitstreamId)
        : model(std::move(model)), channelMapping(std::move(channelMapping)),
          bitstreamId(std4us::fromStd(bitstreamId)) {}

    ProbeSettings(ProbeModel model, std::vector<ChannelIdx> channelMapping,
                  std4us::Optional<BitstreamId> bitstreamId)
        : model(std::move(model)), channelMapping(std::move(channelMapping)),
          bitstreamId(std::move(bitstreamId)) {}

    const std::vector<ChannelIdx> &getChannelMapping() const {
        return channelMapping;
    }

    const ProbeModel &getModel() const {
        return model;
    }

    std::optional<BitstreamId> getBitstreamId() const { return std4us::toStd(bitstreamId); }
    const std4us::Optional<BitstreamId> &getBitstreamIdNative() const { return bitstreamId; }

private:
    ProbeModel model;
    /** A probe channel mapping to the underlying device. */
    std::vector<ChannelIdx> channelMapping;
    std4us::Optional<BitstreamId> bitstreamId;
};
}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
