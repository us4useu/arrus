#ifndef ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
#define ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H

#include <ostream>
#include <utility>

#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/api/devices/FileSettings.h"

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

    SessionSettings(std::vector<arrus::devices::Us4RSettings> us4Rs, std::vector<arrus::devices::FileSettings> files)
        : us4Rs(std::move(us4Rs)), files(std::move(files)) {}

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

private:
    std::vector<arrus::devices::Us4RSettings> us4Rs;
    std::vector<arrus::devices::FileSettings> files;
};

class SessionSettingsBuilder {
    SessionSettingsBuilder() = default;

    void addUs4R(const arrus::devices::Us4RSettings& us4r) {
        us4Rs.push_back(us4r);
    }

    void addFile(const arrus::devices::FileSettings& file) {
        files.push_back(file);
    }

    SessionSettings build() {
        return SessionSettings(us4Rs, files);
    }

private:
    std::vector<arrus::devices::Us4RSettings> us4Rs;
    std::vector<arrus::devices::FileSettings> files;
};

}// namespace arrus::session

#endif//ARRUS_CORE_API_SESSION_SESSIONSETTINGS_H
