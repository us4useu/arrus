#include "ProbeSettings.h"

#include "arrus/common/format.h"
#include "arrus/core/devices/probe/ProbeModel.h"

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const ProbeSettings &settings) {
    os << "model: " << settings.getModel() << " channelMapping: "
       << toString(settings.getChannelMapping());
    return os;
}

}
