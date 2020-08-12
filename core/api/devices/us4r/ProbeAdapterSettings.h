#ifndef ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H

namespace arrus {
class ProbeAdapterSettings {
    using ChannelAddress = std::pair<Ordinal, ChannelIdx>;
    using ProbeAdapterMapping = std::vector<ChannelAddress>;

private:
    ProbeAdapterMapping mapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_US4R_PROBEADAPTERSETTINGS_H
