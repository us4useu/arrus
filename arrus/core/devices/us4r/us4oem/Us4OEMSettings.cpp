#include "Us4OEMSettings.h"

#include "arrus/core/api/devices/us4r/RxSettings.h"

#include <nson/string>

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const Us4OEMSettings &settings) {
    os << "channelMapping: " << nson::join(settings.getChannelMapping(), ", ");
    // TODO(ARRUS-179)
//       << " rxSettings: " << settings.getRxSettings();
    return os;
}

}
