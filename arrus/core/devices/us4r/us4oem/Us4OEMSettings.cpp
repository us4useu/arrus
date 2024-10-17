#include "Us4OEMSettings.h"

#include "arrus/common/format.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const Us4OEMSettings &settings) {
    os << "channelMapping: " << ::arrus::toString(settings.getChannelMapping());
    // TODO(ARRUS-179)
//       << " rxSettings: " << settings.getRxSettings();
    return os;
}

}
