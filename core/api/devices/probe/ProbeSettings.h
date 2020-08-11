#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H

namespace arrus {
class ProbeSettings {
    // TODO ProbeModel
    // TODO ProbeToAdapterConnection
private:
    /** Probe channel -> Adapte channel mapping. */
    std::vector<ChannelIdx> mapping;
};
}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBESETTINGS_H
