#include <std4us/string.h>

#include "arrus/core/devices/file/FileSettings.h"
#include "arrus/core/devices/probe/ProbeModel.h"
namespace arrus::devices {
std::ostream &operator<<(std::ostream &os, const FileSettings &settings) {
    os << "filepath: " << settings.getFilepath()  << ", "
       << "n frames: " << settings.getNFrames() << ", "
       << "probe model: " << settings.getProbeModel();
    return os;
}
}


