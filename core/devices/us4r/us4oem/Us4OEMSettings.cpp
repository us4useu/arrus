#include "Us4OEMSettings.h"

#include "arrus/common/format.h"
#include "arrus/core/devices/us4r/RxSettings.h"

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const Us4OEMSettings &settings) {
    os << "channelMapping: " << ::arrus::toString(settings.getChannelMapping())
       << " activeChannelGroups: "
       << ::arrus::toString(settings.getActiveChannelGroups())
       << " rxSettings: " << settings.getRxSettings();
    return os;
}

}
