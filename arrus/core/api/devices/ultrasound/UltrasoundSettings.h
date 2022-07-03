#ifndef ARRUS_ARRUS_CORE_API_DEVICES_US4R_ULTRASOUNDSETTINGS_H
#define ARRUS_ARRUS_CORE_API_DEVICES_US4R_ULTRASOUNDSETTINGS_H

#include <utility>
#include <vector>
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "UltrasoundFileSettings.h"

namespace arrus::devices {
/**
 * General settings for ultrasound device.
 */
class UltrasoundSettings {
public:
    UltrasoundSettings(ProbeSettings probeSettings, UltrasoundFileSettings backend)
        : probeSettings(std::move(probeSettings)), fileBackend(std::move(backend)) {}

    const ProbeSettings &getProbeSettings() const { return probeSettings; }
    const std::optional<UltrasoundFileSettings> &getFileBackend() const { return fileBackend; }

private:
    ProbeSettings probeSettings;
    // One of:
    std::optional<UltrasoundFileSettings> fileBackend;
    // TODO UltrasoundUs4RSettings, and all other ultrasound devices (e.g. simulators).
};
}

#endif//ARRUS_ARRUS_CORE_API_DEVICES_US4R_ULTRASOUNDSETTINGS_H
