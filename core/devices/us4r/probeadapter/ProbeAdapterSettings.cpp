#include "ProbeAdapterSettings.h"

#include "arrus/common/format.h"

namespace arrus {

std::ostream &
operator<<(std::ostream &os, const ProbeAdapterSettings &settings) {
    os << "modelId: " << settings.getModelId().getName() << ", "
    << settings.getModelId().getManufacturer()
    << " nChannels: " << settings.getNumberOfChannels();
    os << " channelMapping: ";
    for(auto [module, channel] : settings.getChannelMapping()) {
        os << "(" << module << "," << channel << ")";
    }
    return os;
}

}
