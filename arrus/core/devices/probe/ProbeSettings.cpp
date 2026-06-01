#include "ProbeSettings.h"

#include "arrus/core/devices/probe/ProbeModel.h"

#include <std4us/string.h>
namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const ProbeSettings &settings) {
    os << "model: " << settings.getModel() << " channelMapping: "
       << std4us::join(settings.getChannelMapping(), ", ");
    return os;
}

}
