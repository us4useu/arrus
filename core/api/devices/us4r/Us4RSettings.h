#ifndef ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H

#include <utility>
#include <map>
#include <ostream>

#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::devices {

class Us4RSettings {
public:
    explicit Us4RSettings(
        std::vector<Us4OEMSettings> us4OemSettings,
        std::optional<HVSettings> hvSettings)
        : us4oemSettings(std::move(us4OemSettings)),
          hvSettings(std::move(hvSettings)) {}

    Us4RSettings(
        ProbeAdapterSettings probeAdapterSettings,
        ProbeSettings probeSettings,
        RxSettings rxSettings,
        std::optional<HVSettings> hvSettings)
        : probeAdapterSettings(std::move(probeAdapterSettings)),
          probeSettings(std::move(probeSettings)),
          rxSettings(std::move(rxSettings)),
          hvSettings(std::move(hvSettings)) {}

    [[nodiscard]] const std::vector<Us4OEMSettings> &getUs4OEMSettings() const {
        return us4oemSettings;
    }

    [[nodiscard]] const std::optional<ProbeAdapterSettings> &
    getProbeAdapterSettings() const {
        return probeAdapterSettings;
    }

    [[nodiscard]] const std::optional<ProbeSettings> &getProbeSettings() const {
        return probeSettings;
    }

    [[nodiscard]] const std::optional<RxSettings> &getRxSettings() const {
        return rxSettings;
    }

    [[nodiscard]] const std::optional<HVSettings> &getHVSettings() const {
        return hvSettings;
    }

private:
    /* A list of settings for Us4OEMs.
     * First element configures Us4OEM:0, second: Us4OEM:1, etc. */
    std::vector<Us4OEMSettings> us4oemSettings;
    /** Probe adapter settings. Optional - when not set, at least one
     *  Us4OEMSettings must be set. When is set, the list of Us4OEM
     *  settings should be empty. */
    std::optional<ProbeAdapterSettings> probeAdapterSettings{};
    /** ProbeSettings to set. Optional - when is set, ProbeAdapterSettings also
     * must be available.*/
    std::optional<ProbeSettings> probeSettings{};
    /** Required when no Us4OEM settings are set. */
    std::optional<RxSettings> rxSettings;
    /** Optional (us4r devices may have externally controlled hv suppliers. */
    std::optional<HVSettings> hvSettings;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4RSETTINGS_H