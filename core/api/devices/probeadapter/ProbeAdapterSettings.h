#ifndef ARRUS_CORE_API_DEVICES_ADAPTER_ADAPTERSETTINGS_H
#define ARRUS_CORE_API_DEVICES_ADAPTER_ADAPTERSETTINGS_H

namespace arrus {
class ProbeAdapterSettings {
    using ProbeAdapterMapping = std::vector<std::pair<Ordinal, ChannelIdx>>;

private:
    ProbeAdapterMapping mapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_ADAPTER_ADAPTERSETTINGS_H
