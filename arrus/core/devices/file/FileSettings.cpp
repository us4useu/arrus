#include "arrus/core/devices/file/FileSettings.h"
#include "arrus/core/devices/probe/ProbeModel.h"

#include "arrus/common/format.h"


namespace arrus::devices {
std::ostream &operator<<(std::ostream &os, const FileSettings &settings) {
    os << "filepath: " << settings.getFilepath()  << ", "
       << "n frames: " << settings.getNFrames() << ", "
       << "probe model: " << ::arrus::toString(settings.getProbeModel());
    return os;
}
}


