#include "SessionSettings.h"
#include "arrus/core/devices/us4r/Us4RSettings.h"
#include "arrus/core/devices/file/FileSettings.h"

namespace arrus::session {

std::ostream &
operator<<(std::ostream &os, const SessionSettings &settings) {
    for(auto &us4r: settings.getUs4Rs()) {
        os << "us4Rs: " << us4r;
    }
    os << std::endl;
    for(auto &file: settings.getFiles()) {
        os << "files: " << file;
    }
    return os;
}

}


