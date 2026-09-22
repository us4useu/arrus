#include "ProbeSettings.h"

#include "arrus/core/devices/probe/ProbeModel.h"

#include <nson/string>
namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const ProbeSettings &settings) {
    os << "model: " << settings.getModel() << " channelMapping: "
       << nson::join(settings.getChannelMapping(), ", ");
    return os;
}

}
