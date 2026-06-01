#include "Us4OEMSettings.h"

#include "arrus/core/api/devices/us4r/RxSettings.h"

#include <std4us/string.h>

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const Us4OEMSettings &settings) {
    os << "channelMapping: " << std4us::join(settings.getChannelMapping(), ", ");
    // TODO(ARRUS-179)
//       << " rxSettings: " << settings.getRxSettings();
    return os;
}

}
