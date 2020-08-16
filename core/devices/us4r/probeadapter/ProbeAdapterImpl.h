#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H

#include <arrus/core/api/devices/us4r/Us4OEM.h>
#include <arrus/core/api/devices/us4r/ProbeAdapterSettings.h>
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"

namespace arrus {

class ProbeAdapterImpl : public ProbeAdapter {


private:
    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;

    std::vector<Us4OEM::Handle> us4oems;
    std::vector<ChannelAddress> channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
