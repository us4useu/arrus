#ifndef ARRUS_CORE_API_DEVICES_FILESETTINGS_H
#define ARRUS_CORE_API_DEVICES_FILESETTINGS_H

#include <string>

#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "std4us/StdInterop.hpp"
#include "std4us/String.hpp"

namespace arrus::devices {

class FileSettings {
public:
    FileSettings(const std::string &filepath, size_t nFrames, const ProbeModel &probeModel)
        : filepath(std4us::fromStd(filepath)), nFrames(nFrames), probeModel(probeModel) {}

    FileSettings(std4us::String filepath, size_t nFrames, const ProbeModel &probeModel)
        : filepath(std::move(filepath)), nFrames(nFrames), probeModel(probeModel) {}

    std::string getFilepath() const { return std4us::toStd(filepath); }
    void setFilepath(const std::string &fp) { FileSettings::filepath = std4us::fromStd(fp); }

    const std4us::String &getFilepathNative() const { return filepath; }
    void setFilepathNative(std4us::String fp) { FileSettings::filepath = std::move(fp); }

    size_t getNFrames() const { return nFrames; }
    void setNFrames(size_t n) { FileSettings::nFrames = n; }

    const ProbeModel &getProbeModel() const { return probeModel; }
    void setProbeModel(const ProbeModel &model) { FileSettings::probeModel = model; }

private:
    std4us::String filepath;
    /** deprecated(v0.10.0) */
    size_t nFrames;
    /** deprecated(v0.10.0) */
    ProbeModel probeModel;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_FILESETTINGS_H
