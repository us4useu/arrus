#include "Us4OEMSettings.h"

#include "arrus/common/format.h"
#include "arrus/core/devices/us4r/RxSettings.h"

namespace arrus {

std::ostream &
operator<<(std::ostream &os, const Us4OEMSettings &settings) {
    os << "channelMapping: " << toString(settings.getChannelMapping())
       << " activeChannelGroups: " << toString(settings.getActiveChannelGroups())
       << " rxSettings: " << settings.getRxSettings();
    return os;
}

}
