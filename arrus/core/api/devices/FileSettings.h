#ifndef ARRUS_CORE_API_DEVICES_FILESETTINGS_H
#define ARRUS_CORE_API_DEVICES_FILESETTINGS_H

#include <string>
#include "arrus/core/api/devices/probe/ProbeModel.h"

namespace arrus::devices {

class FileSettings {
public:
    FileSettings(const std::string &filepath, size_t nFrames, const ProbeModel &probeModel)
        : filepath(filepath), nFrames(nFrames), probeModel(probeModel) {}

    const std::string &getFilepath() const { return filepath; }
    void setFilepath(const std::string &fp) { FileSettings::filepath = fp; }

    size_t getNFrames() const { return nFrames; }
    void setNFrames(size_t n) { FileSettings::nFrames = n; }

    const ProbeModel &getProbeModel() const { return probeModel; }
    void setProbeModel(const ProbeModel &model) { FileSettings::probeModel = model; }

private:
    std::string filepath;
    /** deprecated(v0.10.0) */
    size_t nFrames;
    /** deprecated(v0.10.0) */
    ProbeModel probeModel;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_FILESETTINGS_H
