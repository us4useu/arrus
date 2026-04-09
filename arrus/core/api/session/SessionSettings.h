#ifndef ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H

#include <ostream>
#include <utility>

#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/api/devices/FileSettings.h"
#include "arrus/core/api/devices/GpuSettings.h"

namespace arrus::session {
class SessionSettings {
public:
    /**
     * Creates session to communicate with given Us4R system.
     *
     * @deprecated(v0.10.0) please use the SessionSettingsBuilder, and create settings for Ultrasound device.
     */
    explicit SessionSettings(arrus::devices::Us4RSettings us4RSettings) {
        this->us4Rs.push_back(std::move(us4RSettings));
    }

    /**
     * @deprecated(v0.10.0) please use the SessionSettingsBuilder, and create settings for Ultrasound device.
     */
    SessionSettings(std::vector<arrus::devices::Us4RSettings> us4Rs, std::vector<arrus::devices::FileSettings> files)
        : us4Rs(std::move(us4Rs)), files(std::move(files)) {}

    /**
     * @deprecated(v0.10.0) please use the SessionSettingsBuilder, and create settings for Ultrasound device.
     */
    SessionSettings(const std::vector<arrus::devices::Us4RSettings> &us4Rs,
                    const std::vector<arrus::devices::FileSettings> &files,
                    const std::vector<arrus::devices::GpuSettings> &gpus)
        : us4Rs(us4Rs), files(files), gpus(gpus) {}

    const arrus::devices::Us4RSettings &getUs4RSettings(::arrus::devices::Ordinal id) const {return us4Rs.at(id);}

    /**
     * Returns the number of us4Rs in this session settings.
     */
    size_t getNumberOfUs4Rs() const {return us4Rs.size(); }

    /**
     * Returns settings of the first us4R device.
     *
     * @deprecated(v0.10.0) please use the getUs4RSettings(Ordinal i)
     * @return reference to us4R settings
     */
    const arrus::devices::Us4RSettings &getUs4RSettings() const { return getUs4RSettings(0); }

    /**
     * Returns the number of files in this session settings.
     */
    size_t getNumberOfFiles() const {return files.size(); }

    const arrus::devices::FileSettings &getFileSettings(::arrus::devices::Ordinal id) const {return files.at(id);}

    const std::vector<arrus::devices::Us4RSettings> &getUs4Rs() const { return us4Rs; }
    const std::vector<arrus::devices::FileSettings> &getFiles() const { return files; }

    const arrus::devices::GpuSettings &getGpuSettings(::arrus::devices::Ordinal id) const {
        return gpus.at(id);
    }
    /** Returns the number of GPUs configured for this session. */
    size_t getNumberOfGpus() const { return gpus.size(); }

private:
    std::vector<arrus::devices::Us4RSettings> us4Rs;
    std::vector<arrus::devices::FileSettings> files;
    std::vector<arrus::devices::GpuSettings> gpus;
};

class SessionSettingsBuilder {
public:
    SessionSettingsBuilder() = default;

    void addUs4R(const arrus::devices::Us4RSettings& us4r) {
        us4Rs.push_back(us4r);
    }

    void addFile(const arrus::devices::FileSettings& file) {
        files.push_back(file);
    }

    void addGpu(const arrus::devices::GpuSettings& gpu) {
        gpus.push_back(gpu);
    }

    SessionSettings build() {
        return SessionSettings(us4Rs, files, gpus);
    }

private:
    std::vector<arrus::devices::Us4RSettings> us4Rs;
    std::vector<arrus::devices::FileSettings> files;
    std::vector<arrus::devices::GpuSettings> gpus;
};

}// namespace arrus::session

#endif//ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
