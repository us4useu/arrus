#include "SessionSettings.h"
#include "arrus/core/devices/us4r/Us4RSettings.h"

namespace arrus {

std::ostream &
operator<<(std::ostream &os, const SessionSettings &settings) {
    os << "us4RSettings: " << settings.getUs4RSettings();
    return os;
}

}


